/********************************************************************************
 * Attempted Real-Time Depth Inference with HailoRT on older Pi-based SDK
 * 
 *  - Captures frames from a libcamera pipeline via GStreamer
 *  - Passes each frame to a Hailo depth network loaded from .hef
 *  - Displays a pseudo-colored depth map in a window
 * 
 * NOTE: Some HailoRT versions may not have device->configure(...). If you get a
 *       "has no member named 'configure'" error, check older examples or docs
 *       on how to load & configure HEF in that release.
 ********************************************************************************/

#include <hailo/hailort.h>   // May be required for older Hailo versions
#include <hailo/hailort.hpp> // The main C++ API
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// If your HailoRT version places these in the global namespace instead of hailort::..., do:
using ::HAILO_STREAM_INTERFACE_PCIE;
using ::HAILO_FORMAT_TYPE_UINT8;
using ::HAILO_FORMAT_TYPE_FLOAT32;
using ::HAILO_SUCCESS;

// A simplified SCDepth-like post-processing function.
// Modify as needed if your depth model uses a different output scale.
template <typename T>
cv::Mat scdepth_post_process(std::vector<T> &logits, int height, int width)
{
    double minVal, maxVal;

    // Convert raw logits into a float matrix of shape (height x width)
    cv::Mat input(height, width, CV_32F, logits.data());
    cv::Mat output = cv::Mat::zeros(height, width, CV_32F);

    // Example transformation from the scdepth code snippet:
    cv::exp(-input, output);
    output = 1.0f / (1.0f + output);
    output = 1.0f / (output * 10.0f + 0.009f);

    // Normalize to [0..255] for visualization
    cv::minMaxIdx(output, &minVal, &maxVal);
    output.convertTo(output, CV_8U, 255.0 / (maxVal - minVal), -(255.0 * minVal) / (maxVal - minVal));

    // Apply a color map to highlight depth differences
    cv::applyColorMap(output, output, cv::COLORMAP_PLASMA);

    return output;
}

// A helper to parse flags like --net=scdepthv3.hef or -n=scdepthv3.hef
static std::string getCmdOption(int argc, char *argv[], const std::string &longOption, const std::string &shortOption)
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

int main(int argc, char** argv)
{
    // 1. Parse command-line for .hef path
    //    e.g. ./realtime_depth_example --net=scdepthv3.hef
    std::string hef_file = getCmdOption(argc, argv, "--net", "-n");
    if (hef_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " --net=<model.hef>\n";
        return 1;
    }

    // 2. Use a GStreamer pipeline with libcamera to open the camera
    //    Adjust width/height as needed. Make sure OpenCV is built with GStreamer support.
    std::string pipeline =
        "libcamerasrc ! video/x-raw,width=640,height=480,format=BGR ! videoconvert ! appsink";
    cv::VideoCapture capture(pipeline, cv::CAP_GSTREAMER);

    if (!capture.isOpened()) {
        std::cerr << "Failed to open libcamera pipeline via GStreamer.\n";
        return 1;
    }

    // 3. Scan for Hailo PCIe devices
    auto devices_result = hailort::Device::scan_pcie();
    if (!devices_result) {
        std::cerr << "Failed scanning PCIe devices: " << devices_result.status() << std::endl;
        return 1;
    }
    if (devices_result->empty()) {
        std::cerr << "No PCIe Hailo devices found.\n";
        return 1;
    }

    // 4. Create a Hailo device via unique_ptr
    auto device_result = hailort::Device::create_pcie(devices_result.value()[0]);
    if (!device_result) {
        std::cerr << "Failed to create Device: " << device_result.status() << std::endl;
        return 1;
    }

    // Must 'move' the result to avoid copying unique_ptr
    auto device = std::move(device_result.value());

    // 5. Load the HEF
    auto hef = hailort::Hef::create(hef_file);
    if (!hef) {
        std::cerr << "Failed to create HEF from file: " << hef_file
                  << ", error=" << hef.status() << std::endl;
        return 1;
    }

    // 6. Create configure params (may differ in older Hailo versions)
    auto configure_params = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!configure_params) {
        std::cerr << "Failed to create configure_params: "
                  << configure_params.status() << std::endl;
        return 1;
    }

    // 7. Configure the device with the HEF
    //    If your older SDK lacks device->configure, see older examples for the correct call.
    auto network_groups = device->configure(hef.value(), configure_params.value());
    if (!network_groups) {
        std::cerr << "device->configure(...) failed or not found: "
                  << network_groups.status() << std::endl;
        return 1;
    }
    if (network_groups->size() != 1) {
        std::cerr << "Expected exactly 1 network group, got "
                  << network_groups->size() << std::endl;
        return 1;
    }
    auto network_group = network_groups.value()[0];

    // 8. Build the input/output vstreams
    //    We'll assume one input and one output for a depth model
    auto in_vstream_params = network_group->make_input_vstream_params(
        /*quantized=*/true, // uses uint8 
        HAILO_FORMAT_TYPE_UINT8,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, 
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE
    );
    if (!in_vstream_params) {
        std::cerr << "Failed to make_input_vstream_params: "
                  << in_vstream_params.status() << std::endl;
        return 1;
    }

    auto out_vstream_params = network_group->make_output_vstream_params(
        /*quantized=*/false, // we'll read float
        HAILO_FORMAT_TYPE_FLOAT32,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, 
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE
    );
    if (!out_vstream_params) {
        std::cerr << "Failed to make_output_vstream_params: "
                  << out_vstream_params.status() << std::endl;
        return 1;
    }

    auto input_vstreams_res = hailort::VStreamsBuilder::create_input_vstreams(
        *network_group, in_vstream_params.value());
    if (!input_vstreams_res) {
        std::cerr << "Failed creating input vstreams: "
                  << input_vstreams_res.status() << std::endl;
        return 1;
    }
    auto output_vstreams_res = hailort::VStreamsBuilder::create_output_vstreams(
        *network_group, out_vstream_params.value());
    if (!output_vstreams_res) {
        std::cerr << "Failed creating output vstreams: "
                  << output_vstreams_res.status() << std::endl;
        return 1;
    }

    // 9. Extract the actual streams (we assume single in/out)
    auto input_vstreams  = input_vstreams_res.release();
    auto output_vstreams = output_vstreams_res.release();
    auto &input_stream  = input_vstreams[0];
    auto &output_stream = output_vstreams[0];

    // 10. Activate the network group
    auto activated = network_group->activate();
    if (!activated) {
        std::cerr << "Failed to activate network group: " << activated.status() << std::endl;
        return 1;
    }

    // 11. Query shapes for resizing frames, allocating output buffers
    auto in_shape = input_stream.get_info().shape;   // e.g. [height, width, channels]
    int in_height   = static_cast<int>(in_shape.height);
    int in_width    = static_cast<int>(in_shape.width);

    auto out_shape = output_stream.get_info().shape; // e.g. [height, width, channels]
    int out_height   = static_cast<int>(out_shape.height);
    int out_width    = static_cast<int>(out_shape.width);
    // Typically 1 channel if it's a depth model.

    // We'll store the float32 output in a buffer sized to the output frame
    size_t out_frame_size = output_stream.get_frame_size(); // in bytes
    std::vector<float> output_data(out_frame_size / sizeof(float));

    std::cout << "\nStarting real-time depth inference (libcamera). Press ESC to exit.\n" << std::endl;

    while (true) {
        // Read a frame from the libcamera pipeline
        cv::Mat frame_bgr;
        capture >> frame_bgr; // "BGR" frames from the pipeline
        if (frame_bgr.empty()) {
            std::cerr << "Empty frame. Stopping.\n";
            break;
        }

        // Convert BGR -> RGB if needed
        if (frame_bgr.channels() == 3) {
            cv::cvtColor(frame_bgr, frame_bgr, cv::COLOR_BGR2RGB);
        }

        // Resize to match Hailo input shape
        if (frame_bgr.cols != in_width || frame_bgr.rows != in_height) {
            cv::resize(frame_bgr, frame_bgr, cv::Size(in_width, in_height));
        }

        // Send to Hailo (assuming UINT8 input)
        hailo_status status = input_stream.write(
            hailort::MemoryView(frame_bgr.data, frame_bgr.total() * frame_bgr.elemSize())
        );
        if (status != HAILO_SUCCESS) {
            std::cerr << "Error: input_stream.write() failed with " << status << std::endl;
            break;
        }

        // Receive output (float32 logits)
        status = output_stream.read(hailort::MemoryView(output_data.data(), out_frame_size));
        if (status != HAILO_SUCCESS) {
            std::cerr << "Error: output_stream.read() failed with " << status << std::endl;
            break;
        }

        // Post-process the depth map
        cv::Mat depth_map = scdepth_post_process<float>(output_data, out_height, out_width);

        // Show the results
        // Convert our "frame_bgr" (actually now in RGB) back to standard BGR for display
        cv::cvtColor(frame_bgr, frame_bgr, cv::COLOR_RGB2BGR);

        // cv::imshow("Live Feed", frame_bgr);
        // cv::imshow("Depth Map", depth_map);

        // Ensure "Depth Map" is fullscreen
        cv::namedWindow("Depth Map", cv::WINDOW_NORMAL);
        cv::setWindowProperty("Depth Map", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        cv::imshow("Depth Map", depth_map);

        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            std::cout << "ESC pressed. Exiting loop.\n";
            break;
        }
    }

    // Clean up
    capture.release();
    cv::destroyAllWindows();

    std::cout << "Done.\n";
    return 0;
}
