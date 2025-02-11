/********************************************************************************
 * A robust multi-thread approach:
 *   1) A background LiDAR thread continuously reads TF Mini data and stores the
 *      latest distance in an atomic variable.
 *   2) Main thread captures frames from Pi camera, runs SCDepth post-processing
 *      in Hailo, and overlays the LiDAR reading. Displays result full-screen.
 *   3) Press ESC to exit, shutting down the LiDAR thread.
 ********************************************************************************/

#include <hailo/hailort.h>
#include <hailo/hailort.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <iostream>
#include <vector>
#include <string>

// For convenience with some older Hailo versions:
using ::HAILO_STREAM_INTERFACE_PCIE;
using ::HAILO_FORMAT_TYPE_UINT8;
using ::HAILO_FORMAT_TYPE_FLOAT32;
using ::HAILO_SUCCESS;

// -----------------------------------------------------------------------------
// 1. Parse cmd-line flags (e.g. --net=model.hef)
// -----------------------------------------------------------------------------
static std::string getCmdOption(int argc, char *argv[],
                                const std::string &longOption,
                                const std::string &shortOption)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg.find(longOption) == 0) || (arg.find(shortOption) == 0)) {
            size_t eq_pos = arg.find('=') + 1;
            return arg.substr(eq_pos);
        }
    }
    return std::string();
}

// -----------------------------------------------------------------------------
// 2. SCDepth-like post-processing you liked:
//    out = 1 / ( (1/(1+exp(-x)))*10 + 0.009 )
// -----------------------------------------------------------------------------
template <typename T>
cv::Mat scdepth_post_process(std::vector<T> &logits, int height, int width)
{
    cv::Mat input(height, width, CV_32F, logits.data());
    cv::Mat depth_map = cv::Mat::zeros(height, width, CV_32F);

    // Step1: tmp = exp(-x)
    // Step2: tmp = 1 / (1 + tmp)
    // Step3: out = 1 / (tmp*10 + 0.009)
    cv::exp(-input, depth_map);
    depth_map = 1.0f / (1.0f + depth_map);
    depth_map = 1.0f / (depth_map*10.0f + 0.009f);

    // Normalize [0..255]
    double minVal, maxVal;
    cv::minMaxIdx(depth_map, &minVal, &maxVal);
    depth_map.convertTo(depth_map, CV_8U,
                        255.0 / (maxVal - minVal),
                        -(255.0 * minVal) / (maxVal - minVal));

    // Apply color map
    cv::applyColorMap(depth_map, depth_map, cv::COLORMAP_PLASMA);
    return depth_map;
}

// -----------------------------------------------------------------------------
// 3. Open TF Mini LiDAR with short read timeouts
// -----------------------------------------------------------------------------
int open_lidar_port(const char *port)
{
    int fd = open(port, O_RDWR | O_NOCTTY | O_SYNC | O_NONBLOCK);
    if (fd < 0) {
        std::cerr << "Error opening LiDAR serial port " << port << std::endl;
        return -1;
    }

    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "tcgetattr() failed for " << port << std::endl;
        close(fd);
        return -1;
    }

    // Baud rate 115200
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    // 8N1
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;  
    tty.c_cflag &= ~PARENB;  
    tty.c_cflag &= ~CSTOPB;  
    tty.c_cflag &= ~CRTSCTS; 
    tty.c_cflag |= (CLOCAL | CREAD);

    // Raw mode
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~OPOST;

    // Non-block or short block
    tty.c_cc[VMIN]  = 0;  // read returns immediately if no data
    tty.c_cc[VTIME] = 1;  // ~0.1s

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "tcsetattr() failed for " << port << std::endl;
        close(fd);
        return -1;
    }
    return fd;
}

// -----------------------------------------------------------------------------
// 4. Read exactly ONE TF Mini frame => distance cm, or -1 if partial/no data
// -----------------------------------------------------------------------------
int read_one_tfmini_frame(int fd)
{
    unsigned char start[2];
    ssize_t n = read(fd, start, 2);
    if (n < 2) {
        return -1; 
    }
    if (start[0] != 0x59 || start[1] != 0x59) {
        return -1; 
    }

    unsigned char buf[7];
    n = read(fd, buf, 7);
    if (n < 7) {
        return -1; 
    }

    int distance = buf[0] + (buf[1] << 8);
    // optional: int strength = buf[2] + (buf[3] << 8);

    // Checksum
    unsigned int sum = 0x59 + 0x59;
    for (int i = 0; i < 6; i++) {
        sum += buf[i];
    }
    if ((sum & 0xFF) != buf[6]) {
        return -1;
    }
    return distance;
}

// -----------------------------------------------------------------------------
// 5. Globals for LiDAR thread
// -----------------------------------------------------------------------------
std::atomic<int> g_lidar_distance{-1};
std::atomic<bool> g_lidar_running{true};

// -----------------------------------------------------------------------------
// 6. LiDAR Thread: read frames in a loop, store newest distance
// -----------------------------------------------------------------------------
void lidar_thread_func(int fd)
{
    std::cout << "[LiDAR Thread] Started.\n";
    while (g_lidar_running.load()) {
        int d = read_one_tfmini_frame(fd);
        if (d > 0) {
            g_lidar_distance.store(d);
        }
        // ~100Hz => LiDAR can produce frames every 10ms
        usleep(5000); // 5ms sleep to reduce CPU usage
    }
    // Cleanup
    close(fd);
    std::cout << "[LiDAR Thread] Exiting.\n";
}

// -----------------------------------------------------------------------------
// 7. Main: SCDepth inference + overlay LiDAR
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // A) Parse .hef
    std::string hef_file = getCmdOption(argc, argv, "--net", "-n");
    if (hef_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " --net=<model.hef>\n";
        return 1;
    }

    // B) LiDAR: open /dev/ttyAMA0, start background thread
    const char *lidar_port = "/dev/ttyAMA0";
    int lidar_fd = open_lidar_port(lidar_port);
    std::thread lidar_thread;
    if (lidar_fd >= 0) {
        lidar_thread = std::thread(lidar_thread_func, lidar_fd);
    } else {
        std::cerr << "Warning: no LiDAR found; continuing w/o it.\n";
        g_lidar_running = false; 
    }

    // C) Camera
    std::string pipeline =
        "libcamerasrc ! video/x-raw,width=640,height=480,format=BGR ! "
        "videoconvert ! appsink";
    cv::VideoCapture capture(pipeline, cv::CAP_GSTREAMER);
    if (!capture.isOpened()) {
        std::cerr << "Failed to open camera pipeline.\n";
        // Stop LiDAR thread if running
        g_lidar_running = false;
        if (lidar_thread.joinable()) {
            lidar_thread.join();
        }
        return 1;
    }

    // D) Hailo Device + Network
    auto devices_res = hailort::Device::scan_pcie();
    if (!devices_res || devices_res->empty()) {
        std::cerr << "No PCIe Hailo devices found.\n";
        // Cleanup LiDAR
        g_lidar_running = false;
        if (lidar_thread.joinable()) {
            lidar_thread.join();
        }
        return 1;
    }
    auto device_res = hailort::Device::create_pcie(devices_res.value()[0]);
    if (!device_res) {
        std::cerr << "Failed creating Hailo device.\n";
        // Cleanup LiDAR
        g_lidar_running = false;
        if (lidar_thread.joinable()) {
            lidar_thread.join();
        }
        return 1;
    }
    auto device = std::move(device_res.value());

    // E) Load HEF
    auto hef = hailort::Hef::create(hef_file);
    if (!hef) {
        std::cerr << "Failed to create HEF from file: " << hef_file << std::endl;
        // Cleanup LiDAR
        g_lidar_running = false;
        if (lidar_thread.joinable()) {
            lidar_thread.join();
        }
        return 1;
    }

    // F) Configure device
    auto cfg_params = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!cfg_params) {
        std::cerr << "Failed to create configure params.\n";
        // Cleanup LiDAR
        g_lidar_running = false;
        if (lidar_thread.joinable()) {
            lidar_thread.join();
        }
        return 1;
    }
    auto network_groups = device->configure(hef.value(), cfg_params.value());
    if (!network_groups || network_groups->size() != 1) {
        std::cerr << "Error configuring device or unexpected # of networks.\n";
        // Cleanup LiDAR
        g_lidar_running = false;
        if (lidar_thread.joinable()) {
            lidar_thread.join();
        }
        return 1;
    }
    auto network_group = network_groups.value()[0];

    // G) Build i/o vstreams
    auto in_vstream_params = network_group->make_input_vstream_params(
        true,  // quantized => uint8
        HAILO_FORMAT_TYPE_UINT8,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE
    );
    auto out_vstream_params = network_group->make_output_vstream_params(
        false, // float => HAILO_FORMAT_TYPE_FLOAT32
        HAILO_FORMAT_TYPE_FLOAT32,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE
    );
    if (!in_vstream_params || !out_vstream_params) {
        std::cerr << "Failed to create vstream params.\n";
        // Cleanup LiDAR
        g_lidar_running = false;
        if (lidar_thread.joinable()) {
            lidar_thread.join();
        }
        return 1;
    }

    auto in_stream_res = hailort::VStreamsBuilder::create_input_vstreams(
        *network_group, in_vstream_params.value());
    auto out_stream_res = hailort::VStreamsBuilder::create_output_vstreams(
        *network_group, out_vstream_params.value());
    if (!in_stream_res || !out_stream_res) {
        std::cerr << "Failed creating i/o vstreams.\n";
        // Cleanup LiDAR
        g_lidar_running = false;
        if (lidar_thread.joinable()) {
            lidar_thread.join();
        }
        return 1;
    }
    auto input_vstreams  = in_stream_res.release();
    auto output_vstreams = out_stream_res.release();
    auto &input_stream  = input_vstreams[0];
    auto &output_stream = output_vstreams[0];

    // H) Activate
    auto activated = network_group->activate();
    if (!activated) {
        std::cerr << "Failed to activate network group.\n";
        // Cleanup LiDAR
        g_lidar_running = false;
        if (lidar_thread.joinable()) {
            lidar_thread.join();
        }
        return 1;
    }

    // I) Query shapes
    auto in_shape  = input_stream.get_info().shape;
    auto out_shape = output_stream.get_info().shape;
    int in_height  = static_cast<int>(in_shape.height);
    int in_width   = static_cast<int>(in_shape.width);
    int out_height = static_cast<int>(out_shape.height);
    int out_width  = static_cast<int>(out_shape.width);

    // Output buffer
    size_t out_frame_size = output_stream.get_frame_size();
    std::vector<float> output_data(out_frame_size / sizeof(float));

    std::cout << "\nStarting multi-threaded SCDepth + LiDAR. ESC to exit.\n";

    // Main loop
    while (true) {
        // 1) Camera frame
        cv::Mat frame_bgr;
        capture >> frame_bgr;
        if (frame_bgr.empty()) {
            std::cerr << "Empty camera frame. Exiting.\n";
            break;
        }

        // 2) Convert BGR->RGB if model expects RGB
        if (frame_bgr.channels() == 3) {
            cv::cvtColor(frame_bgr, frame_bgr, cv::COLOR_BGR2RGB);
        }

        // 3) Resize
        if (frame_bgr.cols != in_width || frame_bgr.rows != in_height) {
            cv::resize(frame_bgr, frame_bgr, cv::Size(in_width, in_height));
        }

        // 4) Inference
        hailo_status status = input_stream.write(
            hailort::MemoryView(frame_bgr.data, frame_bgr.total() * frame_bgr.elemSize()));
        if (status != HAILO_SUCCESS) {
            std::cerr << "Input stream write error.\n";
            break;
        }

        status = output_stream.read(
            hailort::MemoryView(output_data.data(), out_frame_size));
        if (status != HAILO_SUCCESS) {
            std::cerr << "Output stream read error.\n";
            break;
        }

        // 5) SCDepth post-process
        cv::Mat depth_map = scdepth_post_process<float>(output_data, out_height, out_width);

        // 6) Grab latest LiDAR distance
        int dist_cm = g_lidar_distance.load();
        if (dist_cm > 0) {
            // Overlay text at bottom center
            std::string text = std::to_string(dist_cm) + " cm";
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 1.0;
            int thickness = 2;
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
            int x = (depth_map.cols - textSize.width)/2;
            int y = depth_map.rows - 20;
            cv::putText(depth_map, text, cv::Point(x,y),
                        fontFace, fontScale,
                        cv::Scalar(255,255,255), // white
                        thickness);
        }

        // 7) Display full-screen
        cv::namedWindow("Depth Map", cv::WINDOW_NORMAL);
        cv::setWindowProperty("Depth Map", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        cv::imshow("Depth Map", depth_map);

        // ESC to exit
        int key = cv::waitKey(1);
        if (key == 27) { 
            std::cout << "ESC pressed. Exiting.\n";
            break;
        }
    }

    // Cleanup
    capture.release();
    cv::destroyAllWindows();

    // Stop LiDAR thread
    g_lidar_running = false;
    if (lidar_thread.joinable()) {
        lidar_thread.join();
    }

    std::cout << "Done.\n";
    return 0;
}






// Commented out below is a simpler version that does not make use of the LIDAR.



// /********************************************************************************
//  * Attempted Real-Time Depth Inference with HailoRT on older Pi-based SDK
//  * 
//  *  - Captures frames from a libcamera pipeline via GStreamer
//  *  - Passes each frame to a Hailo depth network loaded from .hef
//  *  - Displays a pseudo-colored depth map in a window
//  * 
//  * NOTE: Some HailoRT versions may not have device->configure(...). If you get a
//  *       "has no member named 'configure'" error, check older examples or docs
//  *       on how to load & configure HEF in that release.
//  ********************************************************************************/

// #include <hailo/hailort.h>   // May be required for older Hailo versions
// #include <hailo/hailort.hpp> // The main C++ API
// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <vector>
// #include <string>

// // If your HailoRT version places these in the global namespace instead of hailort::..., do:
// using ::HAILO_STREAM_INTERFACE_PCIE;
// using ::HAILO_FORMAT_TYPE_UINT8;
// using ::HAILO_FORMAT_TYPE_FLOAT32;
// using ::HAILO_SUCCESS;

// // A simplified SCDepth-like post-processing function.
// // Modify as needed if your depth model uses a different output scale.
// template <typename T>
// cv::Mat scdepth_post_process(std::vector<T> &logits, int height, int width)
// {
//     double minVal, maxVal;

//     // Convert raw logits into a float matrix of shape (height x width)
//     cv::Mat input(height, width, CV_32F, logits.data());
//     cv::Mat output = cv::Mat::zeros(height, width, CV_32F);

//     // Example transformation from the scdepth code snippet:
//     cv::exp(-input, output);
//     output = 1.0f / (1.0f + output);
//     output = 1.0f / (output * 10.0f + 0.009f);

//     // Normalize to [0..255] for visualization
//     cv::minMaxIdx(output, &minVal, &maxVal);
//     output.convertTo(output, CV_8U, 255.0 / (maxVal - minVal), -(255.0 * minVal) / (maxVal - minVal));

//     // Apply a color map to highlight depth differences
//     cv::applyColorMap(output, output, cv::COLORMAP_PLASMA);

//     return output;
// }

// // A helper to parse flags like --net=scdepthv3.hef or -n=scdepthv3.hef
// static std::string getCmdOption(int argc, char *argv[], const std::string &longOption, const std::string &shortOption)
// {
//     for (int i = 1; i < argc; ++i) {
//         std::string arg = argv[i];
//         if ((arg.find(longOption) == 0) || (arg.find(shortOption) == 0)) {
//             size_t eq_pos = arg.find('=') + 1;
//             return arg.substr(eq_pos);
//         }
//     }
//     return std::string();
// }

// int main(int argc, char** argv)
// {
//     // 1. Parse command-line for .hef path
//     //    e.g. ./realtime_depth_example --net=scdepthv3.hef
//     std::string hef_file = getCmdOption(argc, argv, "--net", "-n");
//     if (hef_file.empty()) {
//         std::cerr << "Usage: " << argv[0] << " --net=<model.hef>\n";
//         return 1;
//     }

//     // 2. Use a GStreamer pipeline with libcamera to open the camera
//     //    Adjust width/height as needed. Make sure OpenCV is built with GStreamer support.
//     std::string pipeline =
//         "libcamerasrc ! video/x-raw,width=640,height=480,format=BGR ! videoconvert ! appsink";
//     cv::VideoCapture capture(pipeline, cv::CAP_GSTREAMER);

//     if (!capture.isOpened()) {
//         std::cerr << "Failed to open libcamera pipeline via GStreamer.\n";
//         return 1;
//     }

//     // 3. Scan for Hailo PCIe devices
//     auto devices_result = hailort::Device::scan_pcie();
//     if (!devices_result) {
//         std::cerr << "Failed scanning PCIe devices: " << devices_result.status() << std::endl;
//         return 1;
//     }
//     if (devices_result->empty()) {
//         std::cerr << "No PCIe Hailo devices found.\n";
//         return 1;
//     }

//     // 4. Create a Hailo device via unique_ptr
//     auto device_result = hailort::Device::create_pcie(devices_result.value()[0]);
//     if (!device_result) {
//         std::cerr << "Failed to create Device: " << device_result.status() << std::endl;
//         return 1;
//     }

//     // Must 'move' the result to avoid copying unique_ptr
//     auto device = std::move(device_result.value());

//     // 5. Load the HEF
//     auto hef = hailort::Hef::create(hef_file);
//     if (!hef) {
//         std::cerr << "Failed to create HEF from file: " << hef_file
//                   << ", error=" << hef.status() << std::endl;
//         return 1;
//     }

//     // 6. Create configure params (may differ in older Hailo versions)
//     auto configure_params = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
//     if (!configure_params) {
//         std::cerr << "Failed to create configure_params: "
//                   << configure_params.status() << std::endl;
//         return 1;
//     }

//     // 7. Configure the device with the HEF
//     //    If your older SDK lacks device->configure, see older examples for the correct call.
//     auto network_groups = device->configure(hef.value(), configure_params.value());
//     if (!network_groups) {
//         std::cerr << "device->configure(...) failed or not found: "
//                   << network_groups.status() << std::endl;
//         return 1;
//     }
//     if (network_groups->size() != 1) {
//         std::cerr << "Expected exactly 1 network group, got "
//                   << network_groups->size() << std::endl;
//         return 1;
//     }
//     auto network_group = network_groups.value()[0];

//     // 8. Build the input/output vstreams
//     //    We'll assume one input and one output for a depth model
//     auto in_vstream_params = network_group->make_input_vstream_params(
//         /*quantized=*/true, // uses uint8 
//         HAILO_FORMAT_TYPE_UINT8,
//         HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, 
//         HAILO_DEFAULT_VSTREAM_QUEUE_SIZE
//     );
//     if (!in_vstream_params) {
//         std::cerr << "Failed to make_input_vstream_params: "
//                   << in_vstream_params.status() << std::endl;
//         return 1;
//     }

//     auto out_vstream_params = network_group->make_output_vstream_params(
//         /*quantized=*/false, // we'll read float
//         HAILO_FORMAT_TYPE_FLOAT32,
//         HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, 
//         HAILO_DEFAULT_VSTREAM_QUEUE_SIZE
//     );
//     if (!out_vstream_params) {
//         std::cerr << "Failed to make_output_vstream_params: "
//                   << out_vstream_params.status() << std::endl;
//         return 1;
//     }

//     auto input_vstreams_res = hailort::VStreamsBuilder::create_input_vstreams(
//         *network_group, in_vstream_params.value());
//     if (!input_vstreams_res) {
//         std::cerr << "Failed creating input vstreams: "
//                   << input_vstreams_res.status() << std::endl;
//         return 1;
//     }
//     auto output_vstreams_res = hailort::VStreamsBuilder::create_output_vstreams(
//         *network_group, out_vstream_params.value());
//     if (!output_vstreams_res) {
//         std::cerr << "Failed creating output vstreams: "
//                   << output_vstreams_res.status() << std::endl;
//         return 1;
//     }

//     // 9. Extract the actual streams (we assume single in/out)
//     auto input_vstreams  = input_vstreams_res.release();
//     auto output_vstreams = output_vstreams_res.release();
//     auto &input_stream  = input_vstreams[0];
//     auto &output_stream = output_vstreams[0];

//     // 10. Activate the network group
//     auto activated = network_group->activate();
//     if (!activated) {
//         std::cerr << "Failed to activate network group: " << activated.status() << std::endl;
//         return 1;
//     }

//     // 11. Query shapes for resizing frames, allocating output buffers
//     auto in_shape = input_stream.get_info().shape;   // e.g. [height, width, channels]
//     int in_height   = static_cast<int>(in_shape.height);
//     int in_width    = static_cast<int>(in_shape.width);

//     auto out_shape = output_stream.get_info().shape; // e.g. [height, width, channels]
//     int out_height   = static_cast<int>(out_shape.height);
//     int out_width    = static_cast<int>(out_shape.width);
//     // Typically 1 channel if it's a depth model.

//     // We'll store the float32 output in a buffer sized to the output frame
//     size_t out_frame_size = output_stream.get_frame_size(); // in bytes
//     std::vector<float> output_data(out_frame_size / sizeof(float));

//     std::cout << "\nStarting real-time depth inference (libcamera). Press ESC to exit.\n" << std::endl;

//     while (true) {
//         // Read a frame from the libcamera pipeline
//         cv::Mat frame_bgr;
//         capture >> frame_bgr; // "BGR" frames from the pipeline
//         if (frame_bgr.empty()) {
//             std::cerr << "Empty frame. Stopping.\n";
//             break;
//         }

//         // Convert BGR -> RGB if needed
//         if (frame_bgr.channels() == 3) {
//             cv::cvtColor(frame_bgr, frame_bgr, cv::COLOR_BGR2RGB);
//         }

//         // Resize to match Hailo input shape
//         if (frame_bgr.cols != in_width || frame_bgr.rows != in_height) {
//             cv::resize(frame_bgr, frame_bgr, cv::Size(in_width, in_height));
//         }

//         // Send to Hailo (assuming UINT8 input)
//         hailo_status status = input_stream.write(
//             hailort::MemoryView(frame_bgr.data, frame_bgr.total() * frame_bgr.elemSize())
//         );
//         if (status != HAILO_SUCCESS) {
//             std::cerr << "Error: input_stream.write() failed with " << status << std::endl;
//             break;
//         }

//         // Receive output (float32 logits)
//         status = output_stream.read(hailort::MemoryView(output_data.data(), out_frame_size));
//         if (status != HAILO_SUCCESS) {
//             std::cerr << "Error: output_stream.read() failed with " << status << std::endl;
//             break;
//         }

//         // Post-process the depth map
//         cv::Mat depth_map = scdepth_post_process<float>(output_data, out_height, out_width);

//         // Show the results
//         // Convert our "frame_bgr" (actually now in RGB) back to standard BGR for display
//         cv::cvtColor(frame_bgr, frame_bgr, cv::COLOR_RGB2BGR);

//         // cv::imshow("Live Feed", frame_bgr);
//         // cv::imshow("Depth Map", depth_map);

//         // Ensure "Depth Map" is fullscreen
//         cv::namedWindow("Depth Map", cv::WINDOW_NORMAL);
//         cv::setWindowProperty("Depth Map", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//         cv::imshow("Depth Map", depth_map);

//         int key = cv::waitKey(1);
//         if (key == 27) { // ESC
//             std::cout << "ESC pressed. Exiting loop.\n";
//             break;
//         }
//     }

//     // Clean up
//     capture.release();
//     cv::destroyAllWindows();

//     std::cout << "Done.\n";
//     return 0;
// }
