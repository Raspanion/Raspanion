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
#include <cmath>
#include <algorithm>

// For convenience with some older Hailo versions:
using ::HAILO_STREAM_INTERFACE_PCIE;
using ::HAILO_FORMAT_TYPE_UINT8;
using ::HAILO_FORMAT_TYPE_FLOAT32;
using ::HAILO_SUCCESS;

//-------------------------------------------------------------
// Command-line option parser
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
// DepthNet post‑processing function
// Converts NN logits into a depth map (in meters), applies dynamic 
// calibration (using the laser measurement), and produces a color‐mapped
// depth image. It also outputs the calibrated depth map (before normalization)
// so that per‐pixel velocity calculations can be performed over the ROI 
// (central 90%). The laser reading is used only for calibration.
//-------------------------------------------------------------
template <typename T>
void depthnet_post_process(const std::vector<T> &logits,
                           int height, int width,
                           int lidar_cm,  // LiDAR reading (in cm)
                           cv::Mat &depth_color, 
                           cv::Mat &calib_depth)
{
    // Convert NN output into a float matrix.
    cv::Mat depth_float(height, width, CV_32F, const_cast<T*>(logits.data()));

    // Compute the sigmoid output.
    cv::Mat tmp;
    cv::exp(-depth_float, tmp);
    tmp = 1.0f / (1.0f + tmp);

    // Dynamic calibration using the center pixel (which lies within the ROI).
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

    // Compute calibrated depth (in meters) for each pixel.
    tmp = 1.0f / (tmp * A_dynamic + B_dynamic);

    // Save the calibrated depth map.
    calib_depth = tmp.clone();

    // Define ROI (central 90% of the image, excluding outer 5% margins).
    int margin_x = static_cast<int>(width * 0.05);
    int margin_y = static_cast<int>(height * 0.05);
    cv::Rect roi(margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y);

    // Normalize the full depth map using min/max values computed from the ROI.
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(tmp(roi), &min_val, &max_val, &min_loc, &max_loc);
    cv::Mat normalized;
    if (std::fabs(max_val - min_val) < 1e-6)
        normalized = cv::Mat::zeros(tmp.size(), CV_8U);
    else
        tmp.convertTo(normalized, CV_8U, 255.0 / (max_val - min_val),
                      -255.0 * min_val / (max_val - min_val));
    cv::applyColorMap(normalized, depth_color, cv::COLORMAP_JET);

    // Overlay min and max depth values (from the ROI) onto the color map.
    double fontScale = 0.4;
    int thickness = 1, outline_thickness = 2;
    std::ostringstream oss_min, oss_max;
    oss_min << std::fixed << std::setprecision(1) << min_val << " m";
    oss_max << std::fixed << std::setprecision(1) << max_val << " m";
    cv::Point textOffset(5, -5);
    cv::Point min_text_loc = cv::Point(min_loc.x + margin_x, min_loc.y + margin_y) + textOffset;
    cv::Point max_text_loc = cv::Point(max_loc.x + margin_x, max_loc.y + margin_y) + textOffset;
    cv::putText(depth_color, oss_min.str(), min_text_loc,
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0,0,0), outline_thickness);
    cv::putText(depth_color, oss_min.str(), min_text_loc,
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);
    cv::putText(depth_color, oss_max.str(), max_text_loc,
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0,0,0), outline_thickness);
    cv::putText(depth_color, oss_max.str(), max_text_loc,
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);

    // Overlay the laser measurement (for calibration reference) at the center.
    std::ostringstream oss_laser;
    oss_laser << std::fixed << std::setprecision(1) << laser_m << " m";
    cv::Point laserPos(cx - 30, cy - 10); // adjust offset as needed
    cv::putText(depth_color, oss_laser.str(), laserPos, cv::FONT_HERSHEY_SIMPLEX, fontScale,
                cv::Scalar(0,0,0), outline_thickness);
    cv::putText(depth_color, oss_laser.str(), laserPos, cv::FONT_HERSHEY_SIMPLEX, fontScale,
                cv::Scalar(255,255,255), thickness);
}

//-------------------------------------------------------------
// LiDAR serial port functions
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
        sum += buf[i];
    if ((sum & 0xFF) != buf[6])
        return -1;
    return distance;
}

//-------------------------------------------------------------
// Globals for LiDAR thread
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
// Globals and function for video capture thread
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
        usleep(5000); // adjust sleep as needed
    }
}

//-------------------------------------------------------------
// Main function
//-------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string hef_file = getCmdOption(argc, argv, "--net", "-n");
    if (hef_file.empty())
        hef_file = "../../compiled_Hailo_models/scdepthv3.hef";
    
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

    // Create a full screen window for output.
    cv::namedWindow("Depth & Optical Flow", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Depth & Optical Flow", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    std::cout << "[Main] Starting processing loop. Press ESC to exit.\n";

    // For optical flow: previous downsampled grayscale frame.
    cv::Mat smallPrevGray;

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
            cv::resize(frame, frame, cv::Size(in_width, in_height));
        
        // --- DepthNet Inference ---
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
        cv::Mat depth_color, calib_depth;
        int lidar_cm = g_lidar_distance.load();
        if (lidar_cm <= 0)
            lidar_cm = 100; // fallback if no LiDAR reading
        depthnet_post_process<float>(output_data, out_height, out_width, lidar_cm,
                                     depth_color, calib_depth);
        
        // Define ROI (central 90% of full resolution depth map).
        int margin_x = static_cast<int>(out_width * 0.05);
        int margin_y = static_cast<int>(out_height * 0.05);
        cv::Rect roi(margin_x, margin_y, out_width - 2 * margin_x, out_height - 2 * margin_y);
        
        // --- Optical Flow Computation ---
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat smallGray;
        cv::resize(gray, smallGray, cv::Size(), 0.5, 0.5);  // downsample by 0.5
        if (smallPrevGray.empty())
        {
            smallPrevGray = smallGray.clone();
            continue;
        }
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(smallPrevGray, smallGray, flow,
                                     0.5, 3, 15, 3, 5, 1.2, 0);
        smallPrevGray = smallGray.clone();
        
        // --- LS Fit to Estimate Global Angular Rates (ωx, ωy, ωz) ---
        int step = 10;
        int cx_small = smallGray.cols / 2, cy_small = smallGray.rows / 2;
        double FOV_H = 90.0;
        double f_small = (smallGray.cols / 2.0) / tan((FOV_H * CV_PI / 180.0) / 2.0);
        std::vector<cv::Mat> A_rows;
        std::vector<double> b_vals;
        for (int y = 0; y < flow.rows; y += step)
        {
            for (int x = 0; x < flow.cols; x += step)
            {
                cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
                double u_val = fxy.x;
                double v_val = fxy.y;
                double x_centered = x - cx_small;
                double y_centered = y - cy_small;
                cv::Mat row1 = (cv::Mat_<double>(1, 3) << 0, -f_small, -y_centered);
                cv::Mat row2 = (cv::Mat_<double>(1, 3) << f_small, 0, x_centered);
                A_rows.push_back(row1);
                A_rows.push_back(row2);
                b_vals.push_back(u_val);
                b_vals.push_back(v_val);
            }
        }
        int numRows = A_rows.size();
        cv::Mat A(numRows, 3, CV_64F);
        cv::Mat b(numRows, 1, CV_64F);
        for (int i = 0; i < numRows; i++)
        {
            A_rows[i].copyTo(A.row(i));
            b.at<double>(i, 0) = b_vals[i];
        }
        cv::Mat omega;
        cv::solve(A, b, omega, cv::DECOMP_SVD);
        double omega_x = omega.at<double>(0, 0);
        double omega_y = omega.at<double>(1, 0);
        // double omega_z = omega.at<double>(2, 0);
        
        // --- Velocity Calculation Using Per-Pixel Products Over ROI ---
        // For Vz and Vy, multiply global ω by per-pixel depth and then average.
        cv::Scalar sum_depth = cv::sum(calib_depth(roi));
        double area = roi.width * roi.height;
        double avg_depth = sum_depth[0] / area;
        double Vz = omega_x * avg_depth;
        double Vy = omega_y * avg_depth;
        
        // For Vx, compute the per-pixel product of divergence and depth, then average.
        cv::Mat flow_channels[2];
        cv::split(flow, flow_channels);
        cv::Mat du_dx, dv_dy;
        cv::Sobel(flow_channels[0], du_dx, CV_64F, 1, 0, 3);
        cv::Sobel(flow_channels[1], dv_dy, CV_64F, 0, 1, 3);
        cv::Mat div_flow = du_dx + dv_dy;
        // Upsample divergence to full resolution.
        cv::Mat full_div;
        cv::resize(div_flow, full_div, depth_color.size(), 0, 0, cv::INTER_LINEAR);
        // Convert the ROI from calib_depth to CV_64F for multiplication.
        cv::Mat calib_roi;
        calib_depth(roi).convertTo(calib_roi, CV_64F);
        cv::Mat weighted_div = full_div(roi).mul(calib_roi);
        double sum_weighted_div = cv::sum(weighted_div)[0];
        double Vx = - (sum_weighted_div / area) / 2.0;  // forward motion positive
        
        // --- Overlay Optical Flow Vectors on Depth Map ---
        cv::Mat flowOverlay = depth_color.clone();
        for (int y = 0; y < flow.rows; y += step)
        {
            for (int x = 0; x < flow.cols; x += step)
            {
                cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
                cv::Point pt1(x * 2, y * 2);
                cv::Point pt2(cvRound(x * 2 + fxy.x * 2), cvRound(y * 2 + fxy.y * 2));
                cv::arrowedLine(flowOverlay, pt1, pt2, cv::Scalar(0, 255, 0), 1);
                cv::circle(flowOverlay, pt1, 1, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        // --- Overlay Velocity Text (two-tone small fonts) ---
        std::ostringstream oss_vx, oss_vy, oss_vz;
        oss_vx << "Vx: " << std::fixed << std::setprecision(2) << std::showpos << Vx << " m/s";
        oss_vy << "Vy: " << std::fixed << std::setprecision(2) << std::showpos << Vy << " m/s";
        oss_vz << "Vz: " << std::fixed << std::setprecision(2) << std::showpos << Vz << " m/s";
        int baseX = 10, baseY = 80;
        double speedFontScale = 0.4;
        int speedThickness = 1, speedOutlineThickness = 2;
        cv::putText(flowOverlay, oss_vx.str(), cv::Point(baseX, baseY),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(0,0,0), speedOutlineThickness);
        cv::putText(flowOverlay, oss_vx.str(), cv::Point(baseX, baseY),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(255,255,255), speedThickness);
        cv::putText(flowOverlay, oss_vy.str(), cv::Point(baseX, baseY + 20),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(0,0,0), speedOutlineThickness);
        cv::putText(flowOverlay, oss_vy.str(), cv::Point(baseX, baseY + 20),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(255,255,255), speedThickness);
        cv::putText(flowOverlay, oss_vz.str(), cv::Point(baseX, baseY + 40),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(0,0,0), speedOutlineThickness);
        cv::putText(flowOverlay, oss_vz.str(), cv::Point(baseX, baseY + 40),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(255,255,255), speedThickness);
        
        // --- Display the final output ---
        cv::imshow("Depth & Optical Flow", flowOverlay);
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





// #include <hailo/hailort.h>
// #include <hailo/hailort.hpp>
// #include <opencv2/opencv.hpp>

// #include <atomic>
// #include <thread>
// #include <mutex>
// #include <fcntl.h>
// #include <unistd.h>
// #include <termios.h>
// #include <sys/stat.h>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <cstdlib>
// #include <sstream>
// #include <iomanip>
// #include <cmath>
// #include <algorithm>

// // For convenience with some older Hailo versions:
// using ::HAILO_STREAM_INTERFACE_PCIE;
// using ::HAILO_FORMAT_TYPE_UINT8;
// using ::HAILO_FORMAT_TYPE_FLOAT32;
// using ::HAILO_SUCCESS;

// //-------------------------------------------------------------
// // Command-line option parser
// //-------------------------------------------------------------
// static std::string getCmdOption(int argc, char *argv[],
//                                 const std::string &longOption,
//                                 const std::string &shortOption)
// {
//     for (int i = 1; i < argc; ++i)
//     {
//         std::string arg = argv[i];
//         if ((arg.find(longOption) == 0) || (arg.find(shortOption) == 0))
//         {
//             size_t eq_pos = arg.find('=') + 1;
//             return arg.substr(eq_pos);
//         }
//     }
//     return std::string();
// }

// //-------------------------------------------------------------
// // DepthNet post‑processing function
// // Converts NN logits into a depth map (in meters), applies dynamic 
// // calibration using the laser measurement, and produces a colored 
// // depth map. Only the central 90% (excluding outer 5% margins) is used 
// // to compute the min/max normalization and to integrate the depth 
// // values for velocity calculations. The laser measurement is used 
// // solely for calibration.
// //-------------------------------------------------------------
// template <typename T>
// void depthnet_post_process(const std::vector<T> &logits,
//                            int height, int width,
//                            int lidar_cm,  // LiDAR reading (in cm)
//                            double &avg_depth,
//                            cv::Mat &depth_color)
// {
//     // Convert NN output into a float matrix.
//     cv::Mat depth_float(height, width, CV_32F, const_cast<T*>(logits.data()));

//     // Compute the sigmoid output.
//     cv::Mat tmp;
//     cv::exp(-depth_float, tmp);
//     tmp = 1.0f / (1.0f + tmp);

//     // Dynamic calibration using the center pixel (which lies within the ROI)
//     int cx = width / 2, cy = height / 2;
//     float s_ref = tmp.at<float>(cy, cx);
//     static float A_dynamic = 10.0f;   // initial estimate for A
//     static float B_dynamic = 0.009f;  // initial estimate for B
//     float laser_m = static_cast<float>(lidar_cm) / 100.0f;
//     float F = A_dynamic * s_ref + B_dynamic;
//     float F_target = 1.0f / laser_m;
//     float gamma = 0.01f;  // learning rate
//     float F_new = F * std::pow(F_target / F, gamma);
//     float r = (B_dynamic > 1e-6f) ? (A_dynamic / B_dynamic) : 1.0f;
//     A_dynamic = r * F_new / (r * s_ref + 1.0f);
//     B_dynamic = F_new / (r * s_ref + 1.0f);

//     // Compute calibrated depth (in meters) for each pixel.
//     tmp = 1.0f / (tmp * A_dynamic + B_dynamic);

//     // Compute average depth over the central ROI (exclude outer 5% margins).
//     int margin_x = static_cast<int>(width * 0.05);
//     int margin_y = static_cast<int>(height * 0.05);
//     cv::Rect roi(margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y);
//     avg_depth = cv::mean(tmp(roi))[0];

//     // Normalize the full depth map using min/max values from the ROI.
//     double min_val, max_val;
//     cv::Point min_loc, max_loc;
//     cv::minMaxLoc(tmp(roi), &min_val, &max_val, &min_loc, &max_loc);
//     // Adjust min/max locations to full image coordinates.
//     min_loc.x += margin_x;  min_loc.y += margin_y;
//     max_loc.x += margin_x;  max_loc.y += margin_y;
    
//     cv::Mat normalized;
//     if (std::fabs(max_val - min_val) < 1e-6)
//         normalized = cv::Mat::zeros(tmp.size(), CV_8U);
//     else
//         tmp.convertTo(normalized, CV_8U, 255.0 / (max_val - min_val),
//                       -255.0 * min_val / (max_val - min_val));
//     cv::applyColorMap(normalized, depth_color, cv::COLORMAP_JET);

//     // Overlay the min and max depth values (from within the ROI).
//     double fontScale = 0.4;
//     int thickness = 1, outline_thickness = 2;
//     std::ostringstream oss_min, oss_max;
//     oss_min << std::fixed << std::setprecision(1) << min_val << " m";
//     oss_max << std::fixed << std::setprecision(1) << max_val << " m";
//     cv::Point textOffset(5, -5);
//     cv::Point min_text_loc = min_loc + textOffset;
//     cv::Point max_text_loc = max_loc + textOffset;
//     cv::putText(depth_color, oss_min.str(), min_text_loc,
//                 cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0,0,0), outline_thickness);
//     cv::putText(depth_color, oss_min.str(), min_text_loc,
//                 cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);
//     cv::putText(depth_color, oss_max.str(), max_text_loc,
//                 cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0,0,0), outline_thickness);
//     cv::putText(depth_color, oss_max.str(), max_text_loc,
//                 cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);

//     // Overlay the laser measurement at the center (for calibration reference).
//     std::ostringstream oss_laser;
//     oss_laser << std::fixed << std::setprecision(1) << laser_m << " m";
//     cv::Point laserPos(cx - 30, cy - 10); // adjust offset as needed
//     cv::putText(depth_color, oss_laser.str(), laserPos, cv::FONT_HERSHEY_SIMPLEX, fontScale,
//                 cv::Scalar(0,0,0), outline_thickness);
//     cv::putText(depth_color, oss_laser.str(), laserPos, cv::FONT_HERSHEY_SIMPLEX, fontScale,
//                 cv::Scalar(255,255,255), thickness);
// }

// //-------------------------------------------------------------
// // LiDAR serial port functions
// //-------------------------------------------------------------
// int open_lidar_port(const char *port)
// {
//     int fd = open(port, O_RDWR | O_NOCTTY | O_SYNC | O_NONBLOCK);
//     if (fd < 0)
//     {
//         std::cerr << "Error opening LiDAR serial port " << port << std::endl;
//         return -1;
//     }
//     struct termios tty;
//     if (tcgetattr(fd, &tty) != 0)
//     {
//         std::cerr << "tcgetattr() failed for " << port << std::endl;
//         close(fd);
//         return -1;
//     }
//     cfsetospeed(&tty, B115200);
//     cfsetispeed(&tty, B115200);
//     tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
//     tty.c_cflag &= ~PARENB;
//     tty.c_cflag &= ~CSTOPB;
//     tty.c_cflag &= ~CRTSCTS;
//     tty.c_cflag |= (CLOCAL | CREAD);
//     tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
//     tty.c_iflag &= ~(IXON | IXOFF | IXANY);
//     tty.c_oflag &= ~OPOST;
//     tty.c_cc[VMIN] = 0;
//     tty.c_cc[VTIME] = 1;
//     if (tcsetattr(fd, TCSANOW, &tty) != 0)
//     {
//         std::cerr << "tcsetattr() failed for " << port << std::endl;
//         close(fd);
//         return -1;
//     }
//     return fd;
// }

// int read_one_tfmini_frame(int fd)
// {
//     unsigned char start[2];
//     ssize_t n = read(fd, start, 2);
//     if (n < 2)
//         return -1;
//     if (start[0] != 0x59 || start[1] != 0x59)
//         return -1;
//     unsigned char buf[7];
//     n = read(fd, buf, 7);
//     if (n < 7)
//         return -1;
//     int distance = buf[0] + (buf[1] << 8);
//     unsigned int sum = 0x59 + 0x59;
//     for (int i = 0; i < 6; i++)
//         sum += buf[i];
//     if ((sum & 0xFF) != buf[6])
//         return -1;
//     return distance;
// }

// //-------------------------------------------------------------
// // Globals for LiDAR thread
// //-------------------------------------------------------------
// std::atomic<int> g_lidar_distance{-1};
// std::atomic<bool> g_lidar_running{true};

// void lidar_thread_func(int fd)
// {
//     std::cout << "[LiDAR Thread] Started.\n";
//     while (g_lidar_running.load())
//     {
//         int d = read_one_tfmini_frame(fd);
//         if (d > 0)
//             g_lidar_distance.store(d);
//         usleep(5000); // sleep for 5 ms
//     }
//     close(fd);
//     std::cout << "[LiDAR Thread] Exiting.\n";
// }

// //-------------------------------------------------------------
// // Globals and function for video capture thread
// //-------------------------------------------------------------
// std::mutex frame_mutex;
// cv::Mat globalFrame;
// std::atomic<bool> capture_running{true};

// void capture_thread_func(cv::VideoCapture &cap)
// {
//     cv::Mat frame;
//     while (capture_running.load())
//     {
//         cap >> frame;
//         if (!frame.empty())
//         {
//             std::lock_guard<std::mutex> lock(frame_mutex);
//             frame.copyTo(globalFrame);
//         }
//         usleep(5000); // adjust sleep as needed
//     }
// }

// //-------------------------------------------------------------
// // Main function
// //-------------------------------------------------------------
// int main(int argc, char** argv)
// {
//     std::string hef_file = getCmdOption(argc, argv, "--net", "-n");
//     if (hef_file.empty())
//         hef_file = "../../compiled_Hailo_models/scdepthv3.hef";
    
//     const char *lidar_port = "/dev/ttyAMA0";
//     int lidar_fd = open_lidar_port(lidar_port);
//     std::thread lidar_thread;
//     if (lidar_fd >= 0)
//         lidar_thread = std::thread(lidar_thread_func, lidar_fd);
//     else
//     {
//         std::cerr << "Warning: no LiDAR found; continuing without it.\n";
//         g_lidar_running = false;
//     }
    
//     std::string fifoPath = "/tmp/libcamera_vid_fifo";
//     unlink(fifoPath.c_str());
//     if (mkfifo(fifoPath.c_str(), 0666) < 0)
//     {
//         perror("mkfifo");
//         return 1;
//     }
    
//     std::string libcamCmd = "libcamera-vid --nopreview --mode 2328:1748 --inline -t 0 --output " + fifoPath + " &";
//     std::cout << "[Main] Starting libcamera-vid with command:\n" << libcamCmd << std::endl;
//     system(libcamCmd.c_str());
    
//     std::string pipeline = "filesrc location=" + fifoPath +
//                            " ! h264parse ! v4l2h264dec ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
//     std::cout << "[Main] Opening VideoCapture with pipeline:\n" << pipeline << std::endl;
//     cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
//     if (!cap.isOpened())
//     {
//         std::cerr << "Failed to open video capture from FIFO.\n";
//         g_lidar_running = false;
//         if (lidar_thread.joinable())
//             lidar_thread.join();
//         return 1;
//     }
//     std::thread captureThread(capture_thread_func, std::ref(cap));
    
//     auto devices_res = hailort::Device::scan_pcie();
//     if (!devices_res || devices_res->empty())
//     {
//         std::cerr << "No PCIe Hailo devices found.\n";
//         capture_running = false;
//         g_lidar_running = false;
//         if (captureThread.joinable())
//             captureThread.join();
//         if (lidar_thread.joinable())
//             lidar_thread.join();
//         return 1;
//     }
//     auto device_res = hailort::Device::create_pcie(devices_res.value()[0]);
//     if (!device_res)
//     {
//         std::cerr << "Failed creating Hailo device.\n";
//         capture_running = false;
//         g_lidar_running = false;
//         if (captureThread.joinable())
//             captureThread.join();
//         if (lidar_thread.joinable())
//             lidar_thread.join();
//         return 1;
//     }
//     auto device = std::move(device_res.value());
//     auto hef_obj = hailort::Hef::create(hef_file);
//     if (!hef_obj)
//     {
//         std::cerr << "Failed to create HEF from file: " << hef_file << std::endl;
//         capture_running = false;
//         g_lidar_running = false;
//         if (captureThread.joinable())
//             captureThread.join();
//         if (lidar_thread.joinable())
//             lidar_thread.join();
//         return 1;
//     }
//     auto cfg_params = hef_obj->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
//     if (!cfg_params)
//     {
//         std::cerr << "Failed to create configure params.\n";
//         capture_running = false;
//         g_lidar_running = false;
//         if (captureThread.joinable())
//             captureThread.join();
//         if (lidar_thread.joinable())
//             lidar_thread.join();
//         return 1;
//     }
//     auto network_groups = device->configure(hef_obj.value(), cfg_params.value());
//     if (!network_groups || network_groups->size() != 1)
//     {
//         std::cerr << "Error configuring device or unexpected number of networks.\n";
//         capture_running = false;
//         g_lidar_running = false;
//         if (captureThread.joinable())
//             captureThread.join();
//         if (lidar_thread.joinable())
//             lidar_thread.join();
//         return 1;
//     }
//     auto network_group = network_groups.value()[0];
//     auto in_vstream_params = network_group->make_input_vstream_params(
//         true, HAILO_FORMAT_TYPE_UINT8,
//         HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
//         HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
//     auto out_vstream_params = network_group->make_output_vstream_params(
//         false, HAILO_FORMAT_TYPE_FLOAT32,
//         HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
//         HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
//     if (!in_vstream_params || !out_vstream_params)
//     {
//         std::cerr << "Failed to create vstream params.\n";
//         capture_running = false;
//         g_lidar_running = false;
//         if (captureThread.joinable())
//             captureThread.join();
//         if (lidar_thread.joinable())
//             lidar_thread.join();
//         return 1;
//     }
//     auto in_stream_res = hailort::VStreamsBuilder::create_input_vstreams(
//         *network_group, in_vstream_params.value());
//     auto out_stream_res = hailort::VStreamsBuilder::create_output_vstreams(
//         *network_group, out_vstream_params.value());
//     if (!in_stream_res || !out_stream_res)
//     {
//         std::cerr << "Failed creating i/o vstreams.\n";
//         capture_running = false;
//         g_lidar_running = false;
//         if (captureThread.joinable())
//             captureThread.join();
//         if (lidar_thread.joinable())
//             lidar_thread.join();
//         return 1;
//     }
//     auto input_vstreams  = in_stream_res.release();
//     auto output_vstreams = out_stream_res.release();
//     auto &input_stream  = input_vstreams[0];
//     auto &output_stream = output_vstreams[0];
//     auto activated = network_group->activate();
//     if (!activated)
//     {
//         std::cerr << "Failed to activate network group.\n";
//         capture_running = false;
//         g_lidar_running = false;
//         if (captureThread.joinable())
//             captureThread.join();
//         if (lidar_thread.joinable())
//             lidar_thread.join();
//         return 1;
//     }
//     auto in_shape  = input_stream.get_info().shape;
//     auto out_shape = output_stream.get_info().shape;
//     int in_width  = static_cast<int>(in_shape.width);
//     int in_height = static_cast<int>(in_shape.height);
//     int out_width  = static_cast<int>(out_shape.width);
//     int out_height = static_cast<int>(out_shape.height);
//     size_t out_frame_size = output_stream.get_frame_size();
//     std::vector<float> output_data(out_frame_size / sizeof(float));

//     // Create a full screen window for output.
//     cv::namedWindow("Depth & Optical Flow", cv::WINDOW_NORMAL);
//     cv::setWindowProperty("Depth & Optical Flow", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//     std::cout << "[Main] Starting processing loop. Press ESC to exit.\n";

//     // For optical flow: previous downsampled grayscale frame.
//     cv::Mat smallPrevGray;

//     while (true)
//     {
//         cv::Mat frame;
//         {
//             std::lock_guard<std::mutex> lock(frame_mutex);
//             if (!globalFrame.empty())
//                 globalFrame.copyTo(frame);
//         }
//         if (frame.empty())
//         {
//             cv::waitKey(1);
//             continue;
//         }
//         // Resize frame if needed to match network input size.
//         if (frame.cols != in_width || frame.rows != in_height)
//             cv::resize(frame, frame, cv::Size(in_width, in_height));
        
//         // --- DepthNet Inference ---
//         hailo_status status = input_stream.write(
//             hailort::MemoryView(frame.data, frame.total() * frame.elemSize()));
//         if (status != HAILO_SUCCESS)
//         {
//             std::cerr << "Input stream write error.\n";
//             break;
//         }
//         status = output_stream.read(
//             hailort::MemoryView(output_data.data(), out_frame_size));
//         if (status != HAILO_SUCCESS)
//         {
//             std::cerr << "Output stream read error.\n";
//             break;
//         }
//         cv::Mat depth_map;
//         int lidar_cm = g_lidar_distance.load();
//         if (lidar_cm <= 0)
//             lidar_cm = 100; // fallback if no LiDAR reading
//         double avg_depth = 0.0;
//         depthnet_post_process<float>(output_data, out_height, out_width, lidar_cm,
//                                      avg_depth, depth_map);
        
//         // --- Optical Flow Computation ---
//         cv::Mat gray;
//         cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
//         cv::Mat smallGray;
//         cv::resize(gray, smallGray, cv::Size(), 0.5, 0.5);  // downsample by 0.5
        
//         if (smallPrevGray.empty())
//         {
//             smallPrevGray = smallGray.clone();
//             continue; // need two frames to compute flow
//         }
        
//         cv::Mat flow;
//         cv::calcOpticalFlowFarneback(smallPrevGray, smallGray, flow,
//                                      0.5, 3, 15, 3, 5, 1.2, 0);
//         smallPrevGray = smallGray.clone();
        
//         // --- Least Squares to Estimate Angular Rates ---
//         int step = 10;
//         int cx_small = smallGray.cols / 2, cy_small = smallGray.rows / 2;
//         double FOV_H = 90.0; // horizontal FOV (degrees)
//         double f_small = (smallGray.cols / 2.0) / tan((FOV_H * CV_PI / 180.0) / 2.0);
//         std::vector<cv::Mat> A_rows;
//         std::vector<double> b_vals;
//         for (int y = 0; y < flow.rows; y += step)
//         {
//             for (int x = 0; x < flow.cols; x += step)
//             {
//                 cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
//                 double u_val = fxy.x;
//                 double v_val = fxy.y;
//                 double x_centered = x - cx_small;
//                 double y_centered = y - cy_small;
//                 cv::Mat row1 = (cv::Mat_<double>(1, 3) << 0, -f_small, -y_centered);
//                 cv::Mat row2 = (cv::Mat_<double>(1, 3) << f_small, 0, x_centered);
//                 A_rows.push_back(row1);
//                 A_rows.push_back(row2);
//                 b_vals.push_back(u_val);
//                 b_vals.push_back(v_val);
//             }
//         }
//         int numRows = A_rows.size();
//         cv::Mat A(numRows, 3, CV_64F);
//         cv::Mat b(numRows, 1, CV_64F);
//         for (int i = 0; i < numRows; i++)
//         {
//             A_rows[i].copyTo(A.row(i));
//             b.at<double>(i, 0) = b_vals[i];
//         }
//         cv::Mat omega;
//         cv::solve(A, b, omega, cv::DECOMP_SVD);
//         double omega_x = omega.at<double>(0, 0);  // pitch rate (rad/s)
//         double omega_y = omega.at<double>(1, 0);  // yaw rate (rad/s)
//         // double omega_z = omega.at<double>(2, 0);  // roll rate (rad/s)
        
//         // --- Compute Linear Speeds ---
//         // Instead of using a single center depth value, use the average depth
//         // (integrated over the ROI) to compute the velocities.
//         double Vz = omega_x * avg_depth;
//         double Vy = omega_y * avg_depth;
//         std::vector<cv::Mat> flow_channels(2);
//         cv::split(flow, flow_channels);
//         cv::Mat du_dx, dv_dy;
//         cv::Sobel(flow_channels[0], du_dx, CV_64F, 1, 0, 3);
//         cv::Sobel(flow_channels[1], dv_dy, CV_64F, 0, 1, 3);
//         cv::Mat div_flow = du_dx + dv_dy;
//         double avg_div = cv::mean(div_flow)[0];
//         double Vx = (avg_div * avg_depth) / 2.0;  // forward positive
        
//         // --- Overlay Optical Flow on Depth Map ---
//         cv::Mat flowOverlay = depth_map.clone();
//         for (int y = 0; y < flow.rows; y += step)
//         {
//             for (int x = 0; x < flow.cols; x += step)
//             {
//                 cv::Point2f fxy = flow.at<cv::Point2f>(y, x);
//                 cv::Point pt1(x * 2, y * 2);
//                 cv::Point pt2(cvRound(x * 2 + fxy.x * 2), cvRound(y * 2 + fxy.y * 2));
//                 cv::arrowedLine(flowOverlay, pt1, pt2, cv::Scalar(0, 255, 0), 1);
//                 cv::circle(flowOverlay, pt1, 1, cv::Scalar(0, 0, 255), -1);
//             }
//         }
        
//         // --- Overlay Speed Text (smaller two-tone fonts) ---
//         std::ostringstream oss_vx, oss_vy, oss_vz;
//         oss_vx << "Vx: " << std::fixed << std::setprecision(2) << std::showpos << Vx << " m/s";
//         oss_vy << "Vy: " << std::fixed << std::setprecision(2) << std::showpos << Vy << " m/s";
//         oss_vz << "Vz: " << std::fixed << std::setprecision(2) << std::showpos << Vz << " m/s";
//         int baseX = 10, baseY = 80;
//         double speedFontScale = 0.4;
//         int speedThickness = 1, speedOutlineThickness = 2;
//         cv::putText(flowOverlay, oss_vx.str(), cv::Point(baseX, baseY),
//                     cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(0,0,0), speedOutlineThickness);
//         cv::putText(flowOverlay, oss_vx.str(), cv::Point(baseX, baseY),
//                     cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(255,255,255), speedThickness);
//         cv::putText(flowOverlay, oss_vy.str(), cv::Point(baseX, baseY + 20),
//                     cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(0,0,0), speedOutlineThickness);
//         cv::putText(flowOverlay, oss_vy.str(), cv::Point(baseX, baseY + 20),
//                     cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(255,255,255), speedThickness);
//         cv::putText(flowOverlay, oss_vz.str(), cv::Point(baseX, baseY + 40),
//                     cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(0,0,0), speedOutlineThickness);
//         cv::putText(flowOverlay, oss_vz.str(), cv::Point(baseX, baseY + 40),
//                     cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(255,255,255), speedThickness);
        
//         // --- Display the final output ---
//         cv::imshow("Depth & Optical Flow", flowOverlay);
//         if (cv::waitKey(1) == 27)
//         {
//             std::cout << "ESC pressed. Exiting loop.\n";
//             break;
//         }
//     }
//     capture_running = false;
//     if (captureThread.joinable())
//         captureThread.join();
//     cap.release();
//     cv::destroyAllWindows();
//     g_lidar_running = false;
//     if (lidar_thread.joinable())
//         lidar_thread.join();
//     unlink(fifoPath.c_str());
//     std::cout << "[Main] Done.\n";
//     return 0;
// }


