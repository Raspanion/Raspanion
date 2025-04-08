#include <hailo/hailort.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
#include <unistd.h>      // For usleep and unlink
#include <sys/stat.h>    // For mkfifo

//-------------------------------------------------------------
// Global variables for thread-safe frame sharing
//-------------------------------------------------------------
std::mutex frameMutex;
cv::Mat globalFrame;
std::atomic<bool> captureRunning{true};

//-------------------------------------------------------------
// Video Capture Thread Function
//-------------------------------------------------------------
void captureThreadFunc(cv::VideoCapture &cap)
{
    cv::Mat frame;
    while (captureRunning.load()) {
        cap >> frame;
        if (!frame.empty()) {
            std::lock_guard<std::mutex> lock(frameMutex);
            frame.copyTo(globalFrame);
        }
        usleep(5000); // Sleep for 5000 microseconds (5ms)
    }
}

int main()
{
    // HEF file path for Hailo network
    std::string hef_path = "/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef";

    // Create VDevice
    auto vdevice = hailort::VDevice::create();
    if (!vdevice) {
        std::cerr << "Failed to create VDevice\n";
        return 1;
    }

    // Load HEF
    auto hef = hailort::Hef::create(hef_path);
    if (!hef) {
        std::cerr << "Failed to load HEF from " << hef_path << "\n";
        return 1;
    }

    // Create configuration params and configure network
    auto config_params = hef.value().create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
    if (!config_params) {
        std::cerr << "Failed to create configure params\n";
        return 1;
    }

    auto network_groups = vdevice.value()->configure(hef.value(), config_params.value());
    if (!network_groups || network_groups->empty()) {
        std::cerr << "Failed to configure network\n";
        return 1;
    }

    std::cout << "Network configured successfully. Number of groups: " 
              << network_groups.value().size() << std::endl;

    // Setup FIFO for libcamera-vid
    std::string fifoPath = "/tmp/libcamera_vid_fifo";
    unlink(fifoPath.c_str()); // Remove FIFO if already exists
    if (mkfifo(fifoPath.c_str(), 0666) < 0) {
        perror("mkfifo");
        return 1;
    }

    // Launch libcamera-vid with the FIFO output in the background
    std::string libcamCmd = "libcamera-vid --nopreview --mode 2328:1748 --inline -t 0 --output " 
                              + fifoPath + " &";
    std::cout << "[Main] Starting libcamera-vid with command:\n" 
              << libcamCmd << std::endl;
    system(libcamCmd.c_str());

    // Define the GStreamer pipeline for VideoCapture from FIFO
    std::string pipeline = "filesrc location=" + fifoPath +
                           " ! h264parse ! v4l2h264dec ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
    std::cout << "[Main] Opening VideoCapture with pipeline:\n" 
              << pipeline << std::endl;
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video capture from FIFO.\n";
        return 1;
    }

    // Start the capture thread
    std::thread captureThread(captureThreadFunc, std::ref(cap));

    // Create a window for display
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);

    // Main loop: display video frames until user presses 'q' or 'ESC'
    while (true) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!globalFrame.empty()) {
                frame = globalFrame.clone();
            }
        }
        if (!frame.empty()) {
            cv::imshow("Video", frame);
        }

        // Wait 1ms for a key press
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 27) {  // Exit on 'q' key or ESC key (27)
            break;
        }
    }

    // Signal the capture thread to stop and join it
    captureRunning = false;
    if (captureThread.joinable()) {
        captureThread.join();
    }

    // Cleanup: release resources and remove FIFO file
    cap.release();
    cv::destroyAllWindows();
    unlink(fifoPath.c_str());

    return 0;
}



    
    




// #include <hailo/hailort.hpp>
// #include <opencv2/opencv.hpp>
// #include <iostream>

// int main()
// {
//     std::string hef_path = "/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef";

//     // Create VDevice
//     auto vdevice = hailort::VDevice::create();
//     if (!vdevice) {
//         std::cerr << "Failed to create VDevice\n";
//         return 1;
//     }

//     // Load HEF
//     auto hef = hailort::Hef::create(hef_path);
//     if (!hef) {
//         std::cerr << "Failed to load HEF from " << hef_path << "\n";
//         return 1;
//     }

//     // Create config params and configure network
//     auto config_params = hef.value().create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
//     if (!config_params) {
//         std::cerr << "Failed to create configure params\n";
//         return 1;
//     }

//     auto network_groups = vdevice.value()->configure(hef.value(), config_params.value());
//     if (!network_groups || network_groups->empty()) {
//         std::cerr << "Failed to configure network\n";
//         return 1;
//     }

//     std::cout << "Network configured successfully. Number of groups: " << network_groups.value().size() << std::endl;

//     // Camera sanity check
//     cv::VideoCapture cap(0);
//     if (!cap.isOpened()) {
//         std::cerr << "Failed to open camera\n";
//         return 1;
//     }

//     cv::Mat frame;
//     cap >> frame;
//     std::cout << "Captured frame size: " << frame.cols << "x" << frame.rows << std::endl;

//     return 0;
// }
