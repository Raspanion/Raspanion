/********************************************************************************
 * Multi-threaded SCDepth + LiDAR using libcamera-vid for full FOV capture:
 *
 *   - A LiDAR thread continuously reads from the TF Mini and stores the latest
 *     distance in an atomic variable.
 *
 *   - A separate capture thread reads frames from a libcamera-vid FIFO (run as a 
 *     background process with a forced sensor mode for full FOV) and updates a 
 *     shared global frame.
 *
 *   - The main thread uses the latest captured frame to run Hailo inference, 
 *     applies SCDepth post‐processing to create a depth map, overlays the LiDAR 
 *     reading and speed estimates (forward Vx, horizontal Vy, and vertical Vz), 
 *     and displays the depth map full screen.
 *
 *   - The depth map uses a blue (near) to red (far) colormap. The laser (assumed
 *     to be pointing at the center pixel) is used as the reference to continuously 
 *     calibrate the depth mapping using:
 *
 *         depth = 1 / (A * s + B)
 *
 *     A multiplicative update (with gamma) is used so that the computed depth at 
 *     the center converges toward the laser reading.
 *
 *   - The forward speed (Vx) is estimated from the temporal change in the average 
 *     depth over the region 10% in from the left and right edges. The vertical speed 
 *     (Vz) is estimated from the temporal change in the difference of lateral displacements 
 *     computed from the top and bottom regions (which are confined to the central 80% width).  
 *     The horizontal speed (Vy) is computed from left/right ROIs located 5% in from the 
 *     left/right edges and covering the central 50% of the vertical extent.
 *
 *   - Each speed is filtered using a simple 1D Kalman filter with adjusted parameters 
 *     to reduce reactivity to disturbances while still recovering quickly.
 *
 *   - Green boxes are drawn around the ROIs used for Vy and Vz.
 *
 *   - All displayed text is drawn with a black outline (drawn first) and then white text 
 *     on top.
 *
 *   - Press ESC to exit.
 ********************************************************************************/

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
//    calibration, estimates speeds Vx, Vy, and Vz (using Kalman filters), and 
//    overlays text labels (with black outlines). Also draws ROI boxes.
//-------------------------------------------------------------
template <typename T>
void scdepth_post_process(const std::vector<T> &logits,
                          int height, int width,
                          int lidar_cm,  // LiDAR reading (in cm)
                          int &min_est_dist, int &max_est_dist,
                          cv::Mat &depth_color)
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

    // --- Step 4: Speed estimates (dt ~ 0.033 sec, ~30 fps) ---
    double dt = 0.033;

    // (a) Horizontal speed (Vy) from left/right ROIs:
    // Define left ROI: starting 5% from left edge, width = 10% of image,
    // and vertically from 25% to 75% (central 50% of height)
    int left_x = static_cast<int>(width * 0.05);
    int left_w = static_cast<int>(width * 0.10);
    int left_y = static_cast<int>(height * 0.25);
    int left_h = static_cast<int>(height * 0.50);
    cv::Rect leftRect(left_x, left_y, left_w, left_h);
    cv::Mat leftROI = tmp(leftRect);
    double leftAvg = cv::sum(leftROI)[0] / leftROI.total();

    // Define right ROI: starting 5% from right edge (i.e. x = 85%),
    // width = 10% of image, same vertical band as left.
    int right_x = static_cast<int>(width * 0.85);
    int right_w = static_cast<int>(width * 0.10);
    int right_y = left_y;
    int right_h = left_h;
    cv::Rect rightRect(right_x, right_y, right_w, right_h);
    cv::Mat rightROI = tmp(rightRect);
    double rightAvg = cv::sum(rightROI)[0] / rightROI.total();

    // For each ROI, compute its center (in pixels) horizontally:
    double left_center_pixel = left_x + left_w / 2.0;
    double right_center_pixel = right_x + right_w / 2.0;
    // Map these pixel coordinates to angles (in radians).
    double angle_left = (left_center_pixel - width / 2.0) * (1.5708 / width);
    double angle_right = (right_center_pixel - width / 2.0) * (1.5708 / width);
    // Convert average depth values to lateral displacements.
    double X_left = leftAvg * std::tan(angle_left);
    double X_right = rightAvg * std::tan(angle_right);
    double current_horizontal_span = X_right - X_left;
    static double prev_horizontal_span = current_horizontal_span;
    double inst_horiz_speed = (current_horizontal_span - prev_horizontal_span) / dt;
    prev_horizontal_span = current_horizontal_span;

    // (b) Forward speed (Vx) from average depth over central region:
    // Define ROI covering 10% in from the left/right edges (central 80% of width)
    int vx_x = static_cast<int>(width * 0.10);
    int vx_w = static_cast<int>(width * 0.80);
    cv::Rect vxRect(vx_x, 0, vx_w, height);
    double avg_depth = cv::mean(tmp(vxRect))[0];
    static double prev_avg_depth = avg_depth;
    double inst_forward_speed = (prev_avg_depth - avg_depth) / dt;
    prev_avg_depth = avg_depth;

    // (c) Vertical speed (Vz) from top and bottom ROIs:
    // Define top ROI: horizontally central 80% (from 10% to 90%), vertically 10% starting 5% from top.
    int top_x = static_cast<int>(width * 0.10);
    int top_w = static_cast<int>(width * 0.80);
    int top_y = static_cast<int>(height * 0.05);
    int top_h = static_cast<int>(height * 0.10);
    cv::Rect topRect(top_x, top_y, top_w, top_h);
    cv::Mat topROI = tmp(topRect);
    double topAvg = cv::sum(topROI)[0] / topROI.total();

    // Define bottom ROI: same horizontal extent, vertically 10% starting 5% from bottom.
    int bottom_y = static_cast<int>(height * 0.85);
    int bottom_h = static_cast<int>(height * 0.10);
    cv::Rect bottomRect(top_x, bottom_y, top_w, bottom_h);
    cv::Mat bottomROI = tmp(bottomRect);
    double bottomAvg = cv::sum(bottomROI)[0] / bottomROI.total();

    // Compute vertical displacement for each ROI.
    double top_center = top_y + top_h / 2.0;
    double bottom_center = bottom_y + bottom_h / 2.0;
    double angle_top = (top_center - height / 2.0) * (1.5708 / height);
    double angle_bottom = (bottom_center - height / 2.0) * (1.5708 / height);
    double Y_top = topAvg * std::tan(angle_top);
    double Y_bottom = bottomAvg * std::tan(angle_bottom);
    double current_vertical_diff = Y_bottom - Y_top;
    static double prev_vertical_diff = current_vertical_diff;
    double inst_vertical_speed = (current_vertical_diff - prev_vertical_diff) / dt;
    prev_vertical_diff = current_vertical_diff;

    // --- Step 4.5: Kalman filtering for speed estimates ---
    // (Adjusting Q and R to be less reactive to disturbances and allow faster recovery.)
    static double kalman_vx = 0.0, P_vx = 1.0;
    static double kalman_vy = 0.0, P_vy = 1.0;
    static double kalman_vz = 0.0, P_vz = 1.0;
    // Increase both Q and R values compared to before to smooth out disturbances
    double Q_vx = 0.01, R_vx = 0.5;
    double Q_vy = 0.01, R_vy = 0.5;
    double Q_vz = 0.01, R_vz = 0.5;

    double filtered_forward_speed = kalman_update(inst_forward_speed, kalman_vx, P_vx, Q_vx, R_vx);
    double filtered_horiz_speed = kalman_update(inst_horiz_speed, kalman_vy, P_vy, Q_vy, R_vy);
    double filtered_vertical_speed = kalman_update(inst_vertical_speed, kalman_vz, P_vz, Q_vz, R_vz);

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
    cv::rectangle(depth_color, leftRect, cv::Scalar(0,255,0), 2);
    cv::rectangle(depth_color, rightRect, cv::Scalar(0,255,0), 2);
    cv::rectangle(depth_color, topRect, cv::Scalar(0,255,0), 2);
    cv::rectangle(depth_color, bottomRect, cv::Scalar(0,255,0), 2);

    // --- Step 7: Overlay text labels with black outlines ---
    // Depth labels:
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

    // Laser reading at center:
    std::ostringstream oss_laser;
    oss_laser << std::fixed << std::setprecision(1) << laser_m << " m";
    int laser_x = width / 2, laser_y = height / 2;
    cv::Point laserPos(laser_x - 30, laser_y - 10);
    cv::putText(depth_color, oss_laser.str(), laserPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0,0,0), 2);
    cv::putText(depth_color, oss_laser.str(), laserPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255,255,255), 1);

    // Speed estimates with fixed width using std::showpos:
    std::ostringstream oss_vx, oss_vy, oss_vz;
    oss_vx << "Vx: " << std::fixed << std::setprecision(2) << std::showpos << filtered_forward_speed << " m/s";
    oss_vy << "Vy: " << std::fixed << std::setprecision(2) << std::showpos << filtered_horiz_speed << " m/s";
    oss_vz << "Vz: " << std::fixed << std::setprecision(2) << std::showpos << filtered_vertical_speed << " m/s";
    cv::putText(depth_color, oss_vx.str(), cv::Point(10, 80),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 2);
    cv::putText(depth_color, oss_vy.str(), cv::Point(10, 100),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 2);
    cv::putText(depth_color, oss_vz.str(), cv::Point(10, 120),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 2);
    cv::putText(depth_color, oss_vx.str(), cv::Point(10, 80),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);
    cv::putText(depth_color, oss_vy.str(), cv::Point(10, 100),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);
    cv::putText(depth_color, oss_vz.str(), cv::Point(10, 120),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);
}

//-------------------------------------------------------------
// 3. LiDAR serial port functions
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

    cv::namedWindow("Depth Map", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Depth Map", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
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
        cv::Mat depth_map;
        int lidar_cm = g_lidar_distance.load();
        if (lidar_cm <= 0)
            lidar_cm = 100; // fallback if no LiDAR reading
        int min_est_dist = 0, max_est_dist = 0;
        scdepth_post_process<float>(output_data, out_height, out_width, lidar_cm,
                                    min_est_dist, max_est_dist, depth_map);
        cv::imshow("Depth Map", depth_map);
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
