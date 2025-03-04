/*
    Description:
      This script performs camera depth estimation and velocity estimation using
      Hailo and OpenCV. It also communicates with an ArduPilot-based flight controller
      via MAVLink by sending a VISION_SPEED_ESTIMATE (ID 103) message each processing loop.
      Additionally, a separate thread receives and prints MAVLink telemetry (heartbeat,
      battery voltage, attitude, altitude). Note that arming/disarming functions have been removed.
      
    Build requirements:
      - Hailo runtime libraries and headers
      - OpenCV libraries
      - MAVLink headers (for example: mavlink/v2.0/common/mavlink.h)
      - POSIX libraries for serial I/O
*/

#include <hailo/hailort.h>
#include <hailo/hailort.hpp>
#include <opencv2/opencv.hpp>

#include <common/mavlink.h> // Make sure the correct MAVLink dialect is in your include path

#include <atomic>
#include <thread>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <chrono>

//-------------------------------------------------------------
// Command‑line option parser
//-------------------------------------------------------------
static std::string getCmdOption(int argc, char *argv[],
                                const std::string &longOpt,
                                const std::string &shortOpt)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg.find(longOpt) == 0) || (arg.find(shortOpt) == 0)) {
            size_t eqPos = arg.find('=') + 1;
            return arg.substr(eqPos);
        }
    }
    return std::string();
}

//-------------------------------------------------------------
// DepthNet post‑processing function
//-------------------------------------------------------------
template <typename T>
void depthnetPostProcess(const std::vector<T> &logits,
                         int height, int width,
                         int lidar_cm,  // LiDAR reading (in cm)
                         cv::Mat &depthColor, 
                         cv::Mat &calibDepth)
{
    // Convert logits to a float depth map.
    cv::Mat depthFloat(height, width, CV_32F, const_cast<T*>(logits.data()));
    cv::Mat sigmoid;
    cv::exp(-depthFloat, sigmoid);
    sigmoid = 1.0f / (1.0f + sigmoid);

    // Dynamic calibration using the center pixel.
    int cx = width / 2, cy = height / 2;
    float sRef = sigmoid.at<float>(cy, cx);
    static float A_dynamic = 10.0f;
    static float B_dynamic = 0.009f;
    float laser_m = static_cast<float>(lidar_cm) / 100.0f;
    float F = A_dynamic * sRef + B_dynamic;
    float F_target = 1.0f / laser_m;
    float gamma = 0.01f;
    float F_new = F * std::pow(F_target / F, gamma);
    float ratio = (B_dynamic > 1e-6f) ? (A_dynamic / B_dynamic) : 1.0f;
    A_dynamic = ratio * F_new / (ratio * sRef + 1.0f);
    B_dynamic = F_new / (ratio * sRef + 1.0f);

    // Calibrate depth (in meters).
    sigmoid = 1.0f / (sigmoid * A_dynamic + B_dynamic);
    calibDepth = sigmoid.clone();

    // Define ROI (central 90%).
    int marginX = static_cast<int>(width * 0.05);
    int marginY = static_cast<int>(height * 0.05);
    cv::Rect roi(marginX, marginY, width - 2 * marginX, height - 2 * marginY);

    // Normalize depth map.
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(sigmoid(roi), &minVal, &maxVal, &minLoc, &maxLoc);
    cv::Mat normalized;
    if (std::fabs(maxVal - minVal) < 1e-6)
        normalized = cv::Mat::zeros(sigmoid.size(), CV_8U);
    else
        sigmoid.convertTo(normalized, CV_8U, 255.0 / (maxVal - minVal),
                           -255.0 * minVal / (maxVal - minVal));
    cv::applyColorMap(normalized, depthColor, cv::COLORMAP_JET);

    // Overlay depth min/max and laser measurement.
    double fontScale = 0.4;
    int thickness = 1, outlineThick = 2;
    std::ostringstream ossMin, ossMax;
    ossMin << std::fixed << std::setprecision(1) << minVal << " m";
    ossMax << std::fixed << std::setprecision(1) << maxVal << " m";
    cv::Point textOffset(5, -5);
    cv::Point minTextLoc = cv::Point(minLoc.x + marginX, minLoc.y + marginY) + textOffset;
    cv::Point maxTextLoc = cv::Point(maxLoc.x + marginX, maxLoc.y + marginY) + textOffset;
    cv::putText(depthColor, ossMin.str(), minTextLoc, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0,0,0), outlineThick);
    cv::putText(depthColor, ossMin.str(), minTextLoc, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);
    cv::putText(depthColor, ossMax.str(), maxTextLoc, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0,0,0), outlineThick);
    cv::putText(depthColor, ossMax.str(), maxTextLoc, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);

    std::ostringstream ossLaser;
    ossLaser << std::fixed << std::setprecision(1) << laser_m << " m";
    cv::Point laserPos(cx - 30, cy - 10);
    cv::putText(depthColor, ossLaser.str(), laserPos, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0,0,0), outlineThick);
    cv::putText(depthColor, ossLaser.str(), laserPos, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), thickness);
}

//-------------------------------------------------------------
// LiDAR serial port functions
//-------------------------------------------------------------
int openLidarPort(const char *port)
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
    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "tcsetattr() failed for " << port << std::endl;
        close(fd);
        return -1;
    }
    return fd;
}

//-------------------------------------------------------------
// MAVLink serial port functions
//-------------------------------------------------------------
int openMavlinkPort(const char *port)
{
    int fd = open(port, O_RDWR | O_NOCTTY | O_SYNC | O_NONBLOCK);
    if (fd < 0) {
        std::cerr << "Error opening MAVLink serial port " << port << std::endl;
        return -1;
    }
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        std::cerr << "tcgetattr() failed for " << port << std::endl;
        close(fd);
        return -1;
    }
    // Set baudrate to 460800 for MAVLink communication.
    cfsetospeed(&tty, B460800);
    cfsetispeed(&tty, B460800);
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
    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        std::cerr << "tcsetattr() failed for " << port << std::endl;
        close(fd);
        return -1;
    }
    return fd;
}

//-------------------------------------------------------------
// LiDAR Thread
//-------------------------------------------------------------
std::atomic<int> g_lidar_distance{-1};
std::atomic<bool> g_lidar_running{true};

void lidarThreadFunc(int fd)
{
    std::cout << "[LiDAR Thread] Started.\n";
    while (g_lidar_running.load()) {
        // Read one tfmini frame (same as in the original script).
        unsigned char start[2];
        ssize_t n = read(fd, start, 2);
        if (n < 2) continue;
        if (start[0] != 0x59 || start[1] != 0x59) continue;
        unsigned char buf[7];
        n = read(fd, buf, 7);
        if (n < 7) continue;
        int distance = buf[0] + (buf[1] << 8);
        unsigned int sum = 0x59 + 0x59;
        for (int i = 0; i < 6; i++)
            sum += buf[i];
        if ((sum & 0xFF) == buf[6])
            g_lidar_distance.store(distance);
        usleep(5000);
    }
    close(fd);
    std::cout << "[LiDAR Thread] Exiting.\n";
}

//-------------------------------------------------------------
// Gyro Thread
//-------------------------------------------------------------
// Gyro configuration constants:
const int ICM42688_ADDR = 0x69;
const unsigned char PWR_MGMT0   = 0x4E;
const unsigned char GYRO_CONFIG0 = 0x4F;
const unsigned char GYRO_Y_HIGH  = 0x25; // High byte for X-axis
const unsigned char GYRO_Y_LOW   = 0x26; // Low byte for X-axis
const unsigned char GYRO_X_HIGH  = 0x27; // High byte for Y-axis
const unsigned char GYRO_X_LOW   = 0x28; // Low byte for Y-axis
const unsigned char GYRO_Z_HIGH  = 0x29; // High byte for Z-axis (yaw)
const unsigned char GYRO_Z_LOW   = 0x2A; // Low byte for Z-axis

std::mutex gyroMutex;
double rollRate = 0.0;
double pitchRate = 0.0;
double yawRate = 0.0;
std::atomic<bool> g_gyro_running{true};

void gyroThreadFunc()
{
    const char *i2cDevice = "/dev/i2c-1";
    int fd = open(i2cDevice, O_RDWR);
    if (fd < 0) {
        std::cerr << "[Gyro Thread] Failed to open I2C device " << i2cDevice << std::endl;
        return;
    }
    if (ioctl(fd, I2C_SLAVE, ICM42688_ADDR) < 0) {
        std::cerr << "[Gyro Thread] Failed to set I2C slave address." << std::endl;
        close(fd);
        return;
    }
    // Configure gyro:
    unsigned char config[2];
    config[0] = PWR_MGMT0;
    config[1] = 0x0F; // Power on gyro and accel in low-noise mode
    if (write(fd, config, 2) != 2)
        std::cerr << "[Gyro Thread] Failed to write PWR_MGMT0" << std::endl;
    config[0] = GYRO_CONFIG0;
    config[1] = 0x06; // Set gyro ODR to 1kHz and full‑scale range to ±2000 dps
    if (write(fd, config, 2) != 2)
        std::cerr << "[Gyro Thread] Failed to write GYRO_CONFIG0" << std::endl;
    usleep(100000); // Wait 100ms for configuration

    const double GYRO_RAW_TO_DPS = 250.0 / 32768.0;
    const double DPS_TO_RAD = M_PI / 180.0;
    while (g_gyro_running.load()) {
        unsigned char reg = GYRO_Y_HIGH;
        if (write(fd, &reg, 1) != 1) {
            std::cerr << "[Gyro Thread] Failed to set register for gyro read." << std::endl;
            break;
        }
        unsigned char data[6];
        if (read(fd, data, 6) != 6) {
            usleep(5000);
            continue;
        }
        int16_t rawGyroX = (data[2] << 8) | data[3];
        int16_t rawGyroY = (data[0] << 8) | data[1];
        int16_t rawGyroZ = (data[4] << 8) | data[5];
        if (rawGyroX & 0x8000) rawGyroX -= 65536;
        if (rawGyroY & 0x8000) rawGyroY -= 65536;
        if (rawGyroZ & 0x8000) rawGyroZ -= 65536;
        // Apply sign inversion as in the original example.
        rollRate = -rawGyroX * GYRO_RAW_TO_DPS * DPS_TO_RAD;
        pitchRate = -rawGyroY * GYRO_RAW_TO_DPS * DPS_TO_RAD;
        yawRate = rawGyroZ * GYRO_RAW_TO_DPS * DPS_TO_RAD;
        usleep(5000);
    }
    close(fd);
    std::cout << "[Gyro Thread] Exiting.\n";
}

//-------------------------------------------------------------
// Video Capture Thread
//-------------------------------------------------------------
std::mutex frameMutex;
cv::Mat globalFrame;
std::atomic<bool> captureRunning{true};

void captureThreadFunc(cv::VideoCapture &cap)
{
    cv::Mat frame;
    while (captureRunning.load()) {
        cap >> frame;
        if (!frame.empty()) {
            std::lock_guard<std::mutex> lock(frameMutex);
            frame.copyTo(globalFrame);
        }
        usleep(5000);
    }
}

//-------------------------------------------------------------
// MAVLink Communication (Read and Print Threads)
//-------------------------------------------------------------

// Global variables for MAVLink state
std::mutex mavlinkStateMutex;
bool heartbeatReceived = false;
float batteryVoltage = 0.0f;
float rollAngle = 0.0f;      // in radians
float relativeAltitude = 0.0f; // in meters

std::atomic<bool> mavlinkRunning{true};

// MAVLink read thread: reads incoming bytes, decodes messages, and updates state.
void mavlinkReadThreadFunc(int fd)
{
    mavlink_message_t msg;
    mavlink_status_t status;
    uint8_t byte;
    while (mavlinkRunning.load()) {
        ssize_t n = read(fd, &byte, 1);
        if (n > 0) {
            if (mavlink_parse_char(MAVLINK_COMM_0, byte, &msg, &status)) {
                // Process heartbeat
                if (msg.msgid == MAVLINK_MSG_ID_HEARTBEAT) {
                    std::lock_guard<std::mutex> lock(mavlinkStateMutex);
                    heartbeatReceived = true;
                }
                // Process system status for battery voltage
                else if (msg.msgid == MAVLINK_MSG_ID_SYS_STATUS) {
                    mavlink_sys_status_t sys_status;
                    mavlink_msg_sys_status_decode(&msg, &sys_status);
                    std::lock_guard<std::mutex> lock(mavlinkStateMutex);
                    batteryVoltage = sys_status.voltage_battery / 1000.0f;
                }
                // Process attitude (roll angle)
                else if (msg.msgid == MAVLINK_MSG_ID_ATTITUDE) {
                    mavlink_attitude_t attitude;
                    mavlink_msg_attitude_decode(&msg, &attitude);
                    std::lock_guard<std::mutex> lock(mavlinkStateMutex);
                    rollAngle = attitude.roll;
                }
                // Process global position (relative altitude)
                else if (msg.msgid == MAVLINK_MSG_ID_GLOBAL_POSITION_INT) {
                    mavlink_global_position_int_t pos;
                    mavlink_msg_global_position_int_decode(&msg, &pos);
                    std::lock_guard<std::mutex> lock(mavlinkStateMutex);
                    relativeAltitude = pos.relative_alt / 1000.0f;
                }
                // (Additional message processing can be added here as needed.)
            }
        }
    }
}

// MAVLink print thread: periodically prints telemetry from MAVLink state.
void mavlinkPrintThreadFunc()
{
    while (mavlinkRunning.load()) {
        {
            std::lock_guard<std::mutex> lock(mavlinkStateMutex);
            std::cout << "MAVLink Telemetry:" << std::endl;
            if (heartbeatReceived)
                std::cout << "  Heartbeat: RECEIVED" << std::endl;
            else
                std::cout << "  Heartbeat: NOT RECEIVED" << std::endl;
            std::cout << "  Battery Voltage: " << batteryVoltage << " V" << std::endl;
            std::cout << "  Roll Angle: " << (rollAngle * 180.0 / M_PI) << " deg" << std::endl;
            std::cout << "  Relative Altitude: " << relativeAltitude << " m" << std::endl;
            std::cout << "-----------------------" << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

//-------------------------------------------------------------
// Send VISION_SPEED_ESTIMATE MAVLink Message
//-------------------------------------------------------------
//
// This function sends a VISION_SPEED_ESTIMATE (message ID 103) with the
// provided velocities (in m/s) to the flight controller.
void sendVisionSpeedEstimate(int fd, float vx, float vy, float vz)
{
    // Get current timestamp in microseconds.
    uint64_t usec = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();

    mavlink_message_t msg;
    // Use your chosen system and component IDs (e.g., 200) for the vision source.
    const uint8_t system_id = 200;
    const uint8_t component_id = 200;
    
    // Dummy covariance: if you have a 3x3 covariance matrix, use 9 elements (all zeros in this example)
    float dummy_cov[9] = {0.0f};
    uint8_t cov_len = 9;

    // Now pack the message with the additional parameters.
    mavlink_msg_vision_speed_estimate_pack(system_id, component_id, &msg,
                                             usec, vx, vy, vz, dummy_cov, cov_len);

    uint8_t buffer[MAVLINK_MAX_PACKET_LEN];
    int len = mavlink_msg_to_send_buffer(buffer, &msg);
    if (write(fd, buffer, len) < 0) {
        std::cerr << "Failed to send VISION_SPEED_ESTIMATE message." << std::endl;
    }
}

//-------------------------------------------------------------
// Main function
//-------------------------------------------------------------
int main(int argc, char** argv)
{
    // Get HEF file path from command-line or use default.
    std::string hefFile = getCmdOption(argc, argv, "--net", "-n");
    if (hefFile.empty())
        hefFile = "../../compiled_Hailo_models/scdepthv3.hef";

    // Open LiDAR port and start LiDAR thread.
    const char *lidarPort = "/dev/ttyAMA0";
    int lidarFd = openLidarPort(lidarPort);
    std::thread lidarThread;
    if (lidarFd >= 0)
        lidarThread = std::thread(lidarThreadFunc, lidarFd);
    else {
        std::cerr << "Warning: no LiDAR found; continuing without it.\n";
        g_lidar_running = false;
    }

    // Start gyro thread.
    std::thread gyroThread(gyroThreadFunc);

    // Setup libcamera video FIFO.
    std::string fifoPath = "/tmp/libcamera_vid_fifo";
    unlink(fifoPath.c_str());
    if (mkfifo(fifoPath.c_str(), 0666) < 0) {
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
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video capture from FIFO.\n";
        captureRunning = false;
        g_lidar_running = false;
        g_gyro_running = false;
        if (lidarThread.joinable()) lidarThread.join();
        if (gyroThread.joinable()) gyroThread.join();
        return 1;
    }
    std::thread captureThread(captureThreadFunc, std::ref(cap));

    // Initialize Hailo device.
    auto devicesRes = hailort::Device::scan_pcie();
    if (!devicesRes || devicesRes->empty()) {
        std::cerr << "No PCIe Hailo devices found.\n";
        captureRunning = false;
        g_lidar_running = false;
        g_gyro_running = false;
        if (captureThread.joinable()) captureThread.join();
        if (lidarThread.joinable()) lidarThread.join();
        if (gyroThread.joinable()) gyroThread.join();
        return 1;
    }
    auto deviceRes = hailort::Device::create_pcie(devicesRes.value()[0]);
    if (!deviceRes) {
        std::cerr << "Failed creating Hailo device.\n";
        captureRunning = false;
        g_lidar_running = false;
        g_gyro_running = false;
        if (captureThread.joinable()) captureThread.join();
        if (lidarThread.joinable()) lidarThread.join();
        if (gyroThread.joinable()) gyroThread.join();
        return 1;
    }
    auto device = std::move(deviceRes.value());
    auto hefObj = hailort::Hef::create(hefFile);
    if (!hefObj) {
        std::cerr << "Failed to create HEF from file: " << hefFile << std::endl;
        captureRunning = false;
        g_lidar_running = false;
        g_gyro_running = false;
        if (captureThread.joinable()) captureThread.join();
        if (lidarThread.joinable()) lidarThread.join();
        if (gyroThread.joinable()) gyroThread.join();
        return 1;
    }
    auto cfgParams = hefObj->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!cfgParams) {
        std::cerr << "Failed to create configure params.\n";
        captureRunning = false;
        g_lidar_running = false;
        g_gyro_running = false;
        if (captureThread.joinable()) captureThread.join();
        if (lidarThread.joinable()) lidarThread.join();
        if (gyroThread.joinable()) gyroThread.join();
        return 1;
    }
    auto networkGroups = device->configure(hefObj.value(), cfgParams.value());
    if (!networkGroups || networkGroups->size() != 1) {
        std::cerr << "Error configuring device or unexpected number of networks.\n";
        captureRunning = false;
        g_lidar_running = false;
        g_gyro_running = false;
        if (captureThread.joinable()) captureThread.join();
        if (lidarThread.joinable()) lidarThread.join();
        if (gyroThread.joinable()) gyroThread.join();
        return 1;
    }
    auto networkGroup = networkGroups.value()[0];
    auto inVStreamParams = networkGroup->make_input_vstream_params(
        true, HAILO_FORMAT_TYPE_UINT8,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    auto outVStreamParams = networkGroup->make_output_vstream_params(
        false, HAILO_FORMAT_TYPE_FLOAT32,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!inVStreamParams || !outVStreamParams) {
        std::cerr << "Failed to create vstream params.\n";
        captureRunning = false;
        g_lidar_running = false;
        g_gyro_running = false;
        if (captureThread.joinable()) captureThread.join();
        if (lidarThread.joinable()) lidarThread.join();
        if (gyroThread.joinable()) gyroThread.join();
        return 1;
    }
    auto inStreamRes = hailort::VStreamsBuilder::create_input_vstreams(
        *networkGroup, inVStreamParams.value());
    auto outStreamRes = hailort::VStreamsBuilder::create_output_vstreams(
        *networkGroup, outVStreamParams.value());
    if (!inStreamRes || !outStreamRes) {
        std::cerr << "Failed creating i/o vstreams.\n";
        captureRunning = false;
        g_lidar_running = false;
        g_gyro_running = false;
        if (captureThread.joinable()) captureThread.join();
        if (lidarThread.joinable()) lidarThread.join();
        if (gyroThread.joinable()) gyroThread.join();
        return 1;
    }
    auto inputVStreams  = inStreamRes.release();
    auto outputVStreams = outStreamRes.release();
    auto &inputStream  = inputVStreams[0];
    auto &outputStream = outputVStreams[0];
    auto activated = networkGroup->activate();
    if (!activated) {
        std::cerr << "Failed to activate network group.\n";
        captureRunning = false;
        g_lidar_running = false;
        g_gyro_running = false;
        if (captureThread.joinable()) captureThread.join();
        if (lidarThread.joinable()) lidarThread.join();
        if (gyroThread.joinable()) gyroThread.join();
        return 1;
    }
    auto inShape  = inputStream.get_info().shape;
    auto outShape = outputStream.get_info().shape;
    int inWidth  = static_cast<int>(inShape.width);
    int inHeight = static_cast<int>(inShape.height);
    int outWidth  = static_cast<int>(outShape.width);
    int outHeight = static_cast<int>(outShape.height);
    size_t outFrameSize = outputStream.get_frame_size();
    std::vector<float> outputData(outFrameSize / sizeof(float));

    cv::namedWindow("Depth & Optical Flow", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Depth & Optical Flow", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    std::cout << "[Main] Starting processing loop. Press ESC to exit.\n";

    // Open MAVLink serial port.
    const char *mavlinkPort = "/dev/ttyAMA4";
    int mavlinkFd = openMavlinkPort(mavlinkPort);
    if (mavlinkFd < 0) {
        std::cerr << "Failed to open MAVLink port. Exiting.\n";
        return 1;
    }
    // Start MAVLink threads.
    std::thread mavlinkReadThread(mavlinkReadThreadFunc, mavlinkFd);
    std::thread mavlinkPrintThread(mavlinkPrintThreadFunc);

    cv::Mat smallPrevGray;
    cv::Mat predictedFlow;

    // Main processing loop.
    while (true) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!globalFrame.empty())
                globalFrame.copyTo(frame);
        }
        if (frame.empty()) {
            cv::waitKey(1);
            continue;
        }
        if (frame.cols != inWidth || frame.rows != inHeight)
            cv::resize(frame, frame, cv::Size(inWidth, inHeight));

        // DepthNet Inference:
        hailo_status status = inputStream.write(
            hailort::MemoryView(frame.data, frame.total() * frame.elemSize()));
        if (status != HAILO_SUCCESS) {
            std::cerr << "Input stream write error.\n";
            break;
        }
        status = outputStream.read(
            hailort::MemoryView(outputData.data(), outFrameSize));
        if (status != HAILO_SUCCESS) {
            std::cerr << "Output stream read error.\n";
            break;
        }
        cv::Mat depthColor, calibDepth;
        int lidar_cm = g_lidar_distance.load();
        if (lidar_cm <= 0)
            lidar_cm = 100;
        depthnetPostProcess(outputData, outHeight, outWidth, lidar_cm,
                            depthColor, calibDepth);

        int marginX = static_cast<int>(outWidth * 0.05);
        int marginY = static_cast<int>(outHeight * 0.05);
        cv::Rect roi(marginX, marginY, outWidth - 2 * marginX, outHeight - 2 * marginY);

        // Optical Flow Computation on downsampled grayscale image.
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat smallGray;
        cv::resize(gray, smallGray, cv::Size(), 0.5, 0.5);
        if (smallPrevGray.empty()) {
            smallPrevGray = smallGray.clone();
            continue;
        }
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(smallPrevGray, smallGray, flow,
                                     0.5, 3, 15, 3, 5, 1.2, 0);
        smallPrevGray = smallGray.clone();

        // Compute f_small for later use.
        double FOV_H = 140.0;
        double f_small = (smallGray.cols / 2.0) / tan((FOV_H * CV_PI / 180.0) / 2.0);

        // --- Compute Predicted Rotational Flow from Gyro (Blue arrows) ---
        {
            std::lock_guard<std::mutex> lock(gyroMutex);
            predictedFlow = cv::Mat(flow.size(), flow.type(), cv::Scalar(0, 0));
            int cx_flow = flow.cols / 2;
            int cy_flow = flow.rows / 2;
            for (int y = 0; y < flow.rows; y += 10) {
                for (int x = 0; x < flow.cols; x += 10) {
                    double u_pred = f_small * yawRate + rollRate * (y - cy_flow);
                    double v_pred = f_small * pitchRate + rollRate * (cx_flow - x);
                    predictedFlow.at<cv::Point2f>(y, x) =
                        cv::Point2f(static_cast<float>(u_pred),
                                    static_cast<float>(v_pred));
                }
            }
        }

        // --- Compute Corrected (Residual) Flow (Measured - Predicted) ---
        cv::Mat correctedFlow = flow - predictedFlow;

        // --- Compute Linear Velocities from Corrected Flow ---
        int step = 10;
        int cx_small = smallGray.cols / 2, cy_small = smallGray.rows / 2;
        std::vector<cv::Mat> A_rows;
        std::vector<double> b_vals;
        for (int y = 0; y < correctedFlow.rows; y += step) {
            for (int x = 0; x < correctedFlow.cols; x += step) {
                cv::Point2f flowVec = correctedFlow.at<cv::Point2f>(y, x);
                double u_val = flowVec.x;
                double v_val = flowVec.y;
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
        for (int i = 0; i < numRows; i++) {
            A_rows[i].copyTo(A.row(i));
            b.at<double>(i, 0) = b_vals[i];
        }
        cv::Mat omega;
        cv::solve(A, b, omega, cv::DECOMP_SVD);
        double omega_x = omega.at<double>(0, 0);
        double omega_y = omega.at<double>(1, 0);
        cv::Scalar sumDepth = cv::sum(calibDepth(roi));
        double area = roi.width * roi.height;
        double avgDepth = sumDepth[0] / area;
        double Vz = omega_x * avgDepth;  // forward/backward velocity (in m, before scaling)
        double Vy = omega_y * avgDepth;  // lateral velocity

        cv::Mat flowChannels[2];
        cv::split(correctedFlow, flowChannels);
        cv::Mat du_dx, dv_dy;
        cv::Sobel(flowChannels[0], du_dx, CV_64F, 1, 0, 3);
        cv::Sobel(flowChannels[1], dv_dy, CV_64F, 0, 1, 3);
        cv::Mat divFlow = du_dx + dv_dy;
        cv::Mat fullDiv;
        cv::resize(divFlow, fullDiv, depthColor.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat calibROI;
        calibDepth(roi).convertTo(calibROI, CV_64F);
        cv::Mat weightedDiv = fullDiv(roi).mul(calibROI);
        double sumWeightedDiv = cv::sum(weightedDiv)[0];
        double Vx = - (sumWeightedDiv / area) / 2.0;

        // --- NEW: Reverse sign of Vx, convert from m/s to cm/s, and filter linear velocities.
        {
            static double filteredVx = 0.0, filteredVy = 0.0, filteredVz = 0.0;
            static double Vx_scalar = 1.0, Vy_scalar = 1.0, Vz_scalar = 1.0;
            const double linearAlpha = 0.15;  // Smoothing factor
            // Reverse Vx's sign.
            Vx = -Vx;
            // Convert velocities to cm/s.
            Vx *= 100.0;
            Vy *= 100.0;
            Vz *= 100.0;
            // Apply exponential filtering.
            filteredVx = (1 - linearAlpha) * filteredVx + linearAlpha * Vx;
            filteredVy = (1 - linearAlpha) * filteredVy + linearAlpha * Vy;
            filteredVz = (1 - linearAlpha) * filteredVz + linearAlpha * Vz;
            Vx = filteredVx * Vx_scalar;
            Vy = filteredVy * Vy_scalar;
            Vz = filteredVz * Vz_scalar;
        }

        // --- Send VISION_SPEED_ESTIMATE message via MAVLink ---
        // Convert computed speeds from cm/s to m/s.
        float vx_mps = static_cast<float>(Vx) / 100.0f;
        float vy_mps = static_cast<float>(Vy) / 100.0f;
        float vz_mps = static_cast<float>(Vz) / 100.0f;
        sendVisionSpeedEstimate(mavlinkFd, vx_mps, vy_mps, vz_mps);

        // --- Visualization ---
        cv::Mat overlay = depthColor.clone();
        // Draw predicted rotational flow (blue arrows).
        for (int y = 0; y < flow.rows; y += step) {
            for (int x = 0; x < flow.cols; x += step) {
                cv::Point pt1(x * 2, y * 2);
                cv::Point2f predVec = predictedFlow.at<cv::Point2f>(y, x);
                cv::Point pt2(cvRound(x * 2 + predVec.x * 2), cvRound(y * 2 + predVec.y * 2));
                cv::arrowedLine(overlay, pt1, pt2, cv::Scalar(255, 0, 0), 3);
            }
        }
        // Draw measured optical flow (green arrows).
        for (int y = 0; y < flow.rows; y += step) {
            for (int x = 0; x < flow.cols; x += step) {
                cv::Point pt1(x * 2, y * 2);
                cv::Point2f measVec = flow.at<cv::Point2f>(y, x);
                cv::Point pt2(cvRound(x * 2 + measVec.x * 2), cvRound(y * 2 + measVec.y * 2));
                cv::arrowedLine(overlay, pt1, pt2, cv::Scalar(0, 255, 0), 1);
                cv::circle(overlay, pt1, 1, cv::Scalar(0, 0, 255), -1);
            }
        }

        // Overlay computed linear velocities (in cm/s).
        std::ostringstream ossVx, ossVy, ossVz;
        ossVx << "Vx: " << std::fixed << std::setprecision(0) << std::showpos << Vx << " cm/s";
        ossVy << "Vy: " << std::fixed << std::setprecision(0) << std::showpos << Vy << " cm/s";
        ossVz << "Vz: " << std::fixed << std::setprecision(0) << std::showpos << Vz << " cm/s";
        int baseX = 10, baseY = 80;
        double speedFontScale = 0.4;
        int speedThickness = 1, speedOutlineThickness = 2;
        cv::putText(overlay, ossVx.str(), cv::Point(baseX, baseY),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(0,0,0), speedOutlineThickness);
        cv::putText(overlay, ossVx.str(), cv::Point(baseX, baseY),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(255,255,255), speedThickness);
        cv::putText(overlay, ossVy.str(), cv::Point(baseX, baseY + 20),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(0,0,0), speedOutlineThickness);
        cv::putText(overlay, ossVy.str(), cv::Point(baseX, baseY + 20),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(255,255,255), speedThickness);
        cv::putText(overlay, ossVz.str(), cv::Point(baseX, baseY + 40),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(0,0,0), speedOutlineThickness);
        cv::putText(overlay, ossVz.str(), cv::Point(baseX, baseY + 40),
                    cv::FONT_HERSHEY_SIMPLEX, speedFontScale, cv::Scalar(255,255,255), speedThickness);

        cv::imshow("Depth & Optical Flow", overlay);
        if (cv::waitKey(1) == 27) {
            std::cout << "ESC pressed. Exiting loop.\n";
            break;
        }
    }

    // Cleanup: stop threads and close ports.
    captureRunning = false;
    if (captureThread.joinable()) captureThread.join();
    cap.release();
    cv::destroyAllWindows();
    g_lidar_running = false;
    if (lidarThread.joinable()) lidarThread.join();
    g_gyro_running = false;
    if (gyroThread.joinable()) gyroThread.join();

    // Stop MAVLink threads.
    mavlinkRunning = false;
    if (mavlinkReadThread.joinable()) mavlinkReadThread.join();
    if (mavlinkPrintThread.joinable()) mavlinkPrintThread.join();
    close(mavlinkFd);

    unlink(fifoPath.c_str());
    std::cout << "[Main] Done.\n";
    return 0;
}
