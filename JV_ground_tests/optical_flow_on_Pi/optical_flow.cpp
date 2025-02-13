// Wrap C headers in extern "C" for proper C linkage in C++
extern "C" {
    #include <unistd.h>
    #include <sys/stat.h>
}

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>

int main() {
    // ----- Set up libcamera-vid capture via FIFO -----
    std::string fifoPath = "/tmp/libcamera_vid_fifo";
    unlink(fifoPath.c_str());
    if (mkfifo(fifoPath.c_str(), 0666) < 0) {
        perror("mkfifo");
        exit(-1);
    }
    
    std::string libcamCmd = "libcamera-vid --nopreview --mode 2328:1748 --inline -t 0 --output " 
                            + fifoPath + " &";
    std::cout << "Starting libcamera-vid with command:\n" << libcamCmd << std::endl;
    system(libcamCmd.c_str());
    
    std::string pipeline = "filesrc location=" + fifoPath +
                           " ! h264parse ! v4l2h264dec ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
    std::cout << "Opening VideoCapture with pipeline:\n" << pipeline << std::endl;
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open VideoCapture with pipeline." << std::endl;
        exit(-1);
    }
    
    // ----- Capture first frame and convert to grayscale -----
    cv::Mat frame, gray, prevGray;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Error: First frame is empty." << std::endl;
        return -1;
    }
    cv::cvtColor(frame, prevGray, cv::COLOR_BGR2GRAY);
    
    while (true) {
        // Grab the new frame and convert it to grayscale
        cap >> frame;
        if (frame.empty())
            break;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // Detect good features to track in the previous frame
        std::vector<cv::Point2f> prevPts;
        cv::goodFeaturesToTrack(prevGray, prevPts, 200, 0.01, 10);
        if (prevPts.empty()) {
            prevGray = gray.clone();
            continue;
        }
        
        // Calculate optical flow to track these points in the current frame
        std::vector<cv::Point2f> nextPts;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prevGray, gray, prevPts, nextPts, status, err);
        
        // Draw red dots for each successfully tracked point
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i])
                cv::circle(frame, nextPts[i], 2, cv::Scalar(0, 0, 255), -1);
        }
        
        cv::imshow("Optical Flow Dots", frame);
        if (cv::waitKey(1) == 27)  // Exit if ESC is pressed
            break;
        
        prevGray = gray.clone();
    }
    
    cap.release();
    cv::destroyAllWindows();
    unlink(fifoPath.c_str());
    
    return 0;
}
