#include <hailo/hailort.h>
#include <hailo/hailort.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <thread>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <cmath>       // for std::pow, std::tan
#include <algorithm>   // for std::max and std::min

// For convenience with some older Hailo versions:
using ::HAILO_STREAM_INTERFACE_PCIE;
using ::HAILO_FORMAT_TYPE_UINT8;
using ::HAILO_FORMAT_TYPE_FLOAT32;
using ::HAILO_SUCCESS;

//-------------------------------------------------------------
// Simple 1D Kalman filter update function
//-------------------------------------------------------------
double kalman_update(double z, double &x, double &P, double Q, double R)
{
    // Prediction step (constant model)
    double x_pred = x;
    double P_pred = P + Q; // increase uncertainty

    // Kalman gain
    double K = P_pred / (P_pred + R);

    // Update
    x = x_pred + K * (z - x_pred);
    P = (1 - K) * P_pred;
    return x;
}

//-------------------------------------------------------------
// 1. Command-line option parser
//-------------------------------------------------------------
static std::string getCmdOption(int argc, char *argv[],
                                const std::string &longOption,
                                const std::string &shortOption)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg.find(longOption) == 0) || (arg.find(shortOption) == 0))
        {
            size_t eq_pos = arg.find('=') + 1;
            return arg.substr(eq_pos);
        }
    }
    return std::string();
}

//-------------------------------------------------------------
// 2. SCDepth post-processing function.
//    Converts NN logits into a depth map (meters), applies dynamic 
//    calibration, estimates speeds (already computed later via optical flow),
//    overlays text labels and draws ROI boxes.
//    Also produces a raw depth (in meters) Mat for further processing.
//-------------------------------------------------------------
template <typename T>
void scdepth_post_process(const std::vector<T> &logits,
                          int height, int width,
                          int lidar_cm,  // LiDAR reading (in cm)
                          int &min_est_dist, int &max_est_dist,
                          cv::Mat &depth_color,
                          cv::Mat &depth_meters)
{
    // Convert NN output into a float matrix.
    cv::Mat depth_float(height, width, CV_32F, const_cast<T*>(logits.data()));

    // --- Step 1: Compute the sigmoid output, s = 1/(1+exp(-x)) ---
    cv::Mat tmp;
    cv::exp(-depth_float, tmp);
    tmp = 1.0f / (1.0f + tmp);  // now tmp holds s

    // --- Step 2: Dynamic calibration ---
    int cx = width / 2, cy = height / 2;
    float s_ref = tmp.at<float>(cy, cx);
    static float A_dynamic = 10.0f;   // initial estimate for A
    static float B_dynamic = 0.009f;  // initial estimate for B
    float laser_m = static_cast<float>(lidar_cm) / 100.0f;
    float F = A_dynamic * s_ref + B_dynamic;
    float F_target = 1.0f / laser_m;
    float gamma = 0.01f;  // learning rate
    float F_new = F * std::pow(F_target / F, gamma);
    float r = (B_dynamic > 1e-6f) ? (A_dynamic / B_dynamic) : 1.0f;
    A_dynamic = r * F_new / (r * s_ref + 1.0f);
    B_dynamic = F_new / (r * s_ref + 1.0f);

    // --- Step 3: Compute depth (in meters) for each pixel ---
    tmp = 1.0f / (tmp * A_dynamic + B_dynamic);
    // Save the raw depth map (in meters) for later use.
    depth_meters = tmp.clone();

    // --- Steps 4: (Speed estimates here are removed, as we'll compute via optical flow) ---

    // --- Step 5: Normalize the depth map and apply the colormap ---
    int norm_margin_x = static_cast<int>(width * 0.10);
    int norm_margin_y = static_cast<int>(height * 0.10);
    cv::Rect normROI(norm_margin_x, norm_margin_y, width - 2 * norm_margin_x, height - 2 * norm_margin_y);
    double roiMin, roiMax;
    cv::Point roiMinLoc, roiMaxLoc;
    cv::minMaxLoc(tmp(normROI), &roiMin, &roiMax, &roiMinLoc, &roiMaxLoc);
    float min_val = static_cast<float>(roiMin);
    float max_val = static_cast<float>(roiMax);
    cv::Point min_loc = roiMinLoc + cv::Point(norm_margin_x, norm_margin_y);
    cv::Point max_loc = roiMaxLoc + cv::Point(norm_margin_x, norm_margin_y);
    min_est_dist = static_cast<int>(min_val * 1000);
    max_est_dist = static_cast<int>(max_val * 1000);
    cv::Mat normalized;
    if (std::fabs(max_val - min_val) < 1e-6)
        normalized = cv::Mat::zeros(tmp.size(), CV_8U);
    else
        tmp.convertTo(normalized, CV_8U, 255.0 / (max_val - min_val),
                      -255.0 * min_val / (max_val - min_val));
    cv::applyColorMap(normalized, depth_color, cv::COLORMAP_JET);

    // --- Step 6: Draw ROI boxes (green) so you know what areas are used ---
    // (These ROI boxes come from the original speed estimation, kept here for reference.)
    cv::Rect leftRect(static_cast<int>(width * 0.05), static_cast<int>(height * 0.25),
                      static_cast<int>(width * 0.10), static_cast<int>(height * 0.50));
    cv::Rect rightRect(static_cast<int>(width * 0.85), static_cast<int>(height * 0.25),
                       static_cast<int>(width * 0.10), static_cast<int>(height * 0.50));
    cv::Rect topRect(static_cast<int>(width * 0.10), static_cast<int>(height * 0.05),
                     static_cast<int>(width * 0.80), static_cast<int>(height * 0.10));
    cv::Rect bottomRect(static_cast<int>(width * 0.10), static_cast<int>(height * 0.85),
                        static_cast<int>(width * 0.80), static_cast<int>(height * 0.10));
    cv::rectangle(depth_color, leftRect, cv::Scalar(0,255,0), 2);
    cv::rectangle(depth_color, rightRect, cv::Scalar(0,255,0), 2);
    cv::rectangle(depth_color, topRect, cv::Scalar(0,255,0), 2);
    cv::rectangle(depth_color, bottomRect, cv::Scalar(0,255,0), 2);

    // --- Step 7: Overlay text labels (for depth and laser reading) ---
    std::ostringstream oss_min, oss_max;
    oss_min << std::fixed << std::setprecision(1) << min_val << " m";
    oss_max << std::fixed << std::setprecision(1) << max_val << " m";
    cv::putText(depth_color, oss_min.str(), min_loc, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0,0,0), 2);
    cv::putText(depth_color, oss_min.str(), min_loc, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255,255,255), 1);
    cv::putText(depth_color, oss_max.str(), max_loc, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0,0,0), 2);
    cv::putText(depth_color, oss_max.str(), max_loc, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255,255,255), 1);

    std::ostringstream oss_laser;
    oss_laser << std::fixed << std::setprecision(1) << laser_m << " m";
    int laser_x = width / 2, laser_y = height / 2;
    cv::Point laserPos(laser_x - 30, laser_y - 10);
    cv::putText(depth_color, oss_laser.str(), laserPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0,0,0), 2);
    cv::putText(depth_color, oss_laser.str(), laserPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255,255,255), 1);
}

//-------------------------------------------------------------
// 3. LiDAR serial port functions (unchanged)
//-------------------------------------------------------------
int open_lidar_port(const char *port)
{
    int fd = open(port, O_RDWR | O_NOCTTY | O_SYNC | O_NONBLOCK);
    if (fd < 0)
    {
        std::cerr << "Error opening LiDAR serial port " << port << std::endl;
        return -1;
    }
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0)
    {
        std::cerr << "tcgetattr() failed for " << port << std::endl;
        close(fd);
        return -1;
    }
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~OPOST;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;
    if (tcsetattr(fd, TCSANOW, &tty) != 0)
    {
        std::cerr << "tcsetattr() failed for " << port << std::endl;
        close(fd);
        return -1;
    }
    return fd;
}

int read_one_tfmini_frame(int fd)
{
    unsigned char start[2];
    ssize_t n = read(fd, start, 2);
    if (n < 2)
        return -1;
    if (start[0] != 0x59 || start[1] != 0x59)
        return -1;
    unsigned char buf[7];
    n = read(fd, buf, 7);
    if (n < 7)
        return -1;
    int distance = buf[0] + (buf[1] << 8);
    unsigned int sum = 0x59 + 0x59;
    for (int i = 0; i < 6; i++)
    {
        sum += buf[i];
    }
    if ((sum & 0xFF) != buf[6])
        return -1;
    return distance;
}

//-------------------------------------------------------------
// 4. Globals for LiDAR thread
//-------------------------------------------------------------
std::atomic<int> g_lidar_distance{-1};
std::atomic<bool> g_lidar_running{true};

void lidar_thread_func(int fd)
{
    std::cout << "[LiDAR Thread] Started.\n";
    while (g_lidar_running.load())
    {
        int d = read_one_tfmini_frame(fd);
        if (d > 0)
            g_lidar_distance.store(d);
        usleep(5000); // sleep for 5 ms
    }
    close(fd);
    std::cout << "[LiDAR Thread] Exiting.\n";
}

//-------------------------------------------------------------
// 5. Globals and function for video capture thread
//-------------------------------------------------------------
std::mutex frame_mutex;
cv::Mat globalFrame;
std::atomic<bool> capture_running{true};

void capture_thread_func(cv::VideoCapture &cap)
{
    cv::Mat frame;
    while (capture_running.load())
    {
        cap >> frame;
        if (!frame.empty())
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame.copyTo(globalFrame);
        }
        usleep(500); // adjust sleep as needed
    }
}

//-------------------------------------------------------------
// 6. Main function
//-------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string hef_file = getCmdOption(argc, argv, "--net", "-n");
    if (hef_file.empty())
    {
        hef_file = "../../compiled_Hailo_models/scdepthv3.hef";
    }
    const char *lidar_port = "/dev/ttyAMA0";
    int lidar_fd = open_lidar_port(lidar_port);
    std::thread lidar_thread;
    if (lidar_fd >= 0)
        lidar_thread = std::thread(lidar_thread_func, lidar_fd);
    else
    {
        std::cerr << "Warning: no LiDAR found; continuing without it.\n";
        g_lidar_running = false;
    }
    std::string fifoPath = "/tmp/libcamera_vid_fifo";
    unlink(fifoPath.c_str());
    if (mkfifo(fifoPath.c_str(), 0666) < 0)
    {
        perror("mkfifo");
        return 1;
    }
    std::string libcamCmd = "libcamera-vid --nopreview --mode 2328:1748 --inline -t 0 --output " + fifoPath + " &";
    std::cout << "[Main] Starting libcamera-vid with command:\n" << libcamCmd << std::endl;
    system(libcamCmd.c_str());
    
    std::string pipeline = "filesrc location=" + fifoPath +
                           " ! h264parse ! v4l2h264dec ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
    std::cout << "[Main] Opening VideoCapture with pipeline:\n" << pipeline << std::endl;
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video capture from FIFO.\n";
        g_lidar_running = false;
        if (lidar_thread.joinable())
            lidar_thread.join();
        return 1;
    }
    std::thread captureThread(capture_thread_func, std::ref(cap));
    
    auto devices_res = hailort::Device::scan_pcie();
    if (!devices_res || devices_res->empty())
    {
        std::cerr << "No PCIe Hailo devices found.\n";
        capture_running = false;
        g_lidar_running = false;
        if (captureThread.joinable())
            captureThread.join();
        if (lidar_thread.joinable())
            lidar_thread.join();
        return 1;
    }
    auto device_res = hailort::Device::create_pcie(devices_res.value()[0]);
    if (!device_res)
    {
        std::cerr << "Failed creating Hailo device.\n";
        capture_running = false;
        g_lidar_running = false;
        if (captureThread.joinable())
            captureThread.join();
        if (lidar_thread.joinable())
            lidar_thread.join();
        return 1;
    }
    auto device = std::move(device_res.value());
    auto hef_obj = hailort::Hef::create(hef_file);
    if (!hef_obj)
    {
        std::cerr << "Failed to create HEF from file: " << hef_file << std::endl;
        capture_running = false;
        g_lidar_running = false;
        if (captureThread.joinable())
            captureThread.join();
        if (lidar_thread.joinable())
            lidar_thread.join();
        return 1;
    }
    auto cfg_params = hef_obj->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!cfg_params)
    {
        std::cerr << "Failed to create configure params.\n";
        capture_running = false;
        g_lidar_running = false;
        if (captureThread.joinable())
            captureThread.join();
        if (lidar_thread.joinable())
            lidar_thread.join();
        return 1;
    }
    auto network_groups = device->configure(hef_obj.value(), cfg_params.value());
    if (!network_groups || network_groups->size() != 1)
    {
        std::cerr << "Error configuring device or unexpected number of networks.\n";
        capture_running = false;
        g_lidar_running = false;
        if (captureThread.joinable())
            captureThread.join();
        if (lidar_thread.joinable())
            lidar_thread.join();
        return 1;
    }
    auto network_group = network_groups.value()[0];
    auto in_vstream_params = network_group->make_input_vstream_params(
        true, HAILO_FORMAT_TYPE_UINT8,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    auto out_vstream_params = network_group->make_output_vstream_params(
        false, HAILO_FORMAT_TYPE_FLOAT32,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!in_vstream_params || !out_vstream_params)
    {
        std::cerr << "Failed to create vstream params.\n";
        capture_running = false;
        g_lidar_running = false;
        if (captureThread.joinable())
            captureThread.join();
        if (lidar_thread.joinable())
            lidar_thread.join();
        return 1;
    }
    auto in_stream_res = hailort::VStreamsBuilder::create_input_vstreams(
        *network_group, in_vstream_params.value());
    auto out_stream_res = hailort::VStreamsBuilder::create_output_vstreams(
        *network_group, out_vstream_params.value());
    if (!in_stream_res || !out_stream_res)
    {
        std::cerr << "Failed creating i/o vstreams.\n";
        capture_running = false;
        g_lidar_running = false;
        if (captureThread.joinable())
            captureThread.join();
        if (lidar_thread.joinable())
            lidar_thread.join();
        return 1;
    }
    auto input_vstreams  = in_stream_res.release();
    auto output_vstreams = out_stream_res.release();
    auto &input_stream  = input_vstreams[0];
    auto &output_stream = output_vstreams[0];
    auto activated = network_group->activate();
    if (!activated)
    {
        std::cerr << "Failed to activate network group.\n";
        capture_running = false;
        g_lidar_running = false;
        if (captureThread.joinable())
            captureThread.join();
        if (lidar_thread.joinable())
            lidar_thread.join();
        return 1;
    }
    auto in_shape  = input_stream.get_info().shape;
    auto out_shape = output_stream.get_info().shape;
    int in_width  = static_cast<int>(in_shape.width);
    int in_height = static_cast<int>(in_shape.height);
    int out_width  = static_cast<int>(out_shape.width);
    int out_height = static_cast<int>(out_shape.height);
    size_t out_frame_size = output_stream.get_frame_size();
    std::vector<float> output_data(out_frame_size / sizeof(float));

    // Create a full screen window for output
    cv::namedWindow("Depth Map", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Depth Map", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    // We'll downscale optical flow images by a factor (same as before)
    double scaleFactor = 0.5;
    // Compute effective focal length for the downscaled image (assuming horizontal FOV=90°)
    double FOV_H = 90.0; // degrees
    double f_small = (in_width * scaleFactor / 2.0) / tan((FOV_H * CV_PI / 180.0) / 2.0);

    // For optical flow, we need a previous grayscale image (downscaled)
    cv::Mat smallPrevGrayFlow;

    std::cout << "[Main] Starting processing loop. Press ESC to exit.\n";
    while (true)
    {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (!globalFrame.empty())
                globalFrame.copyTo(frame);
        }
        if (frame.empty())
        {
            cv::waitKey(1);
            continue;
        }
        if (frame.cols != in_width || frame.rows != in_height)
        {
            cv::resize(frame, frame, cv::Size(in_width, in_height));
        }
        // Write frame to Hailo input stream and get network output.
        hailo_status status = input_stream.write(
            hailort::MemoryView(frame.data, frame.total() * frame.elemSize()));
        if (status != HAILO_SUCCESS)
        {
            std::cerr << "Input stream write error.\n";
            break;
        }
        status = output_stream.read(
            hailort::MemoryView(output_data.data(), out_frame_size));
        if (status != HAILO_SUCCESS)
        {
            std::cerr << "Output stream read error.\n";
            break;
        }
        cv::Mat depth_color;
        cv::Mat depth_meters; // raw depth map (in meters)
        int lidar_cm = g_lidar_distance.load();
        if (lidar_cm <= 0)
            lidar_cm = 100; // fallback if no LiDAR reading
        int min_est_dist = 0, max_est_dist = 0;
        scdepth_post_process<float>(output_data, out_height, out_width, lidar_cm,
                                    min_est_dist, max_est_dist, depth_color, depth_meters);

        // ---- Dense Optical Flow Processing for Speed Estimation ----
        // Convert current frame to grayscale and downscale for optical flow
        cv::Mat currentGray, smallCurrentGray;
        cv::cvtColor(frame, currentGray, cv::COLOR_BGR2GRAY);
        cv::resize(currentGray, smallCurrentGray, cv::Size(), scaleFactor, scaleFactor);
        cv::Mat flow;
        if (!smallPrevGrayFlow.empty())
        {
            // Compute dense optical flow between previous and current downscaled grayscale images
            cv::calcOpticalFlowFarneback(smallPrevGrayFlow, smallCurrentGray, flow,
                                         0.5, 3, 15, 3, 5, 1.2, 0);
            // Downscale the raw depth map to match optical flow image size
            cv::Mat smallDepth;
            cv::resize(depth_meters, smallDepth, smallCurrentGray.size(), 0, 0, cv::INTER_LINEAR);

            int step = 10;
            double sumVz = 0.0, sumVy = 0.0;
            int count = 0;
            // For each sampled pixel, compute "angular" displacements (in radians)
            // then multiply by local depth to get a local linear speed component.
            for (int y = 0; y < flow.rows; y += step)
            {
                for (int x = 0; x < flow.cols; x += step)
                {
                    cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
                    double pitch_rate = fxy.y / f_small; // vertical displacement → pitch (radians)
                    double yaw_rate = fxy.x / f_small;   // horizontal displacement → yaw (radians)
                    float local_depth = smallDepth.at<float>(y, x);
                    sumVz += pitch_rate * local_depth;
                    sumVy += yaw_rate * local_depth;
                    count++;
                }
            }
            double avgVz = (count > 0) ? sumVz / count : 0.0;
            double avgVy = (count > 0) ? sumVy / count : 0.0;

            // Estimate forward speed (Vx) via divergence.
            cv::Mat flow_u, flow_v;
            std::vector<cv::Mat> flow_channels;
            cv::split(flow, flow_channels);
            flow_u = flow_channels[0];
            flow_v = flow_channels[1];
            cv::Mat du_dx, dv_dy;
            cv::Sobel(flow_u, du_dx, CV_64F, 1, 0, 3);
            cv::Sobel(flow_v, dv_dy, CV_64F, 0, 1, 3);
            cv::Mat divergence = du_dx + dv_dy;
            cv::Scalar avg_div = cv::mean(divergence);
            double avg_divergence = avg_div[0];
            cv::Scalar avg_depth_scalar = cv::mean(smallDepth);
            double avg_depth = avg_depth_scalar[0];
            // Assuming divergence ≈ -2*Vx/avg_depth  ⇒  Vx = - (avg_divergence * avg_depth) / 2
            double avgVx = - avg_divergence * avg_depth / 2.0;

            // ---- Create Optical Flow Visualization ----
            cv::Mat flowVis;
            cv::cvtColor(smallCurrentGray, flowVis, cv::COLOR_GRAY2BGR);
            for (int y = 0; y < flow.rows; y += step)
            {
                for (int x = 0; x < flow.cols; x += step)
                {
                    cv::Point2f pt1(x, y);
                    cv::Point2f pt2 = pt1 + flow.at<cv::Point2f>(y, x);
                    cv::arrowedLine(flowVis, pt1, pt2, cv::Scalar(0,255,0), 1, cv::LINE_AA);
                }
            }
            // Upscale the optical flow visualization to match the depth map size.
            cv::Mat flowVisUpscaled;
            cv::resize(flowVis, flowVisUpscaled, depth_color.size());
            // Blend the flow visualization (30% weight) over the depth color image (70% weight).
            cv::Mat finalOutput;
            cv::addWeighted(depth_color, 0.7, flowVisUpscaled, 0.3, 0, finalOutput);

            // ---- Overlay Speed Text Labels ----
            std::ostringstream oss_vx, oss_vy, oss_vz;
            oss_vx << "Vx: " << std::fixed << std::setprecision(2) << std::showpos << avgVx << " m/s";
            oss_vy << "Vy: " << std::fixed << std::setprecision(2) << std::showpos << avgVy << " m/s";
            oss_vz << "Vz: " << std::fixed << std::setprecision(2) << std::showpos << avgVz << " m/s";
            cv::putText(finalOutput, oss_vx.str(), cv::Point(10, 140),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 2);
            cv::putText(finalOutput, oss_vx.str(), cv::Point(10, 140),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);
            cv::putText(finalOutput, oss_vy.str(), cv::Point(10, 160),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 2);
            cv::putText(finalOutput, oss_vy.str(), cv::Point(10, 160),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);
            cv::putText(finalOutput, oss_vz.str(), cv::Point(10, 180),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 2);
            cv::putText(finalOutput, oss_vz.str(), cv::Point(10, 180),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);
            
            cv::imshow("Depth Map", finalOutput);
        }
        else
        {
            // No previous frame for optical flow; simply show the depth image.
            cv::imshow("Depth Map", depth_color);
        }
        // Update the optical flow previous frame.
        smallPrevGrayFlow = smallCurrentGray.clone();

        if (cv::waitKey(1) == 27)
        {
            std::cout << "ESC pressed. Exiting loop.\n";
            break;
        }
    }
    capture_running = false;
    if (captureThread.joinable())
        captureThread.join();
    cap.release();
    cv::destroyAllWindows();
    g_lidar_running = false;
    if (lidar_thread.joinable())
        lidar_thread.join();
    unlink(fifoPath.c_str());
    std::cout << "[Main] Done.\n";
    return 0;
}
