// Wrap C headers in extern "C" for proper C linkage in C++
extern "C" {
    #include <unistd.h>
    #include <sys/stat.h>
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

int main() {
    // ----- Set up libcamera-vid capture via FIFO -----
    string fifoPath = "/tmp/libcamera_vid_fifo";
    unlink(fifoPath.c_str());
    if (mkfifo(fifoPath.c_str(), 0666) < 0) {
        perror("mkfifo");
        exit(-1);
    }
    
    string libcamCmd = "libcamera-vid --nopreview --mode 2328:1748 --inline -t 0 --output " 
                         + fifoPath + " &";
    cout << "Starting libcamera-vid with command:\n" << libcamCmd << endl;
    system(libcamCmd.c_str());
    
    string pipeline = "filesrc location=" + fifoPath +
                      " ! h264parse ! v4l2h264dec ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
    cout << "Opening VideoCapture with pipeline:\n" << pipeline << endl;
    VideoCapture cap(pipeline, CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open VideoCapture with pipeline." << endl;
        exit(-1);
    }
    
    // ----- Grab the first frame and initialize -----
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Error: First frame is empty." << endl;
        return -1;
    }
    
    // Downscale factor (adjust as needed)
    double scaleFactor = 0.5;
    
    // Convert first frame to grayscale and then downscale
    Mat gray, smallGray, smallPrevGray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    resize(gray, smallPrevGray, Size(), scaleFactor, scaleFactor);
    
    // Compute effective focal length for the downscaled image.
    // Assume horizontal FOV of 90°.
    double FOV_H = 90.0; // degrees
    double smallWidth = frame.cols * scaleFactor;
    double f_small = (smallWidth / 2.0) / tan((FOV_H * CV_PI / 180.0) / 2.0);
    
    // Create a full screen window
    namedWindow("Dense Optical Flow with Angular Rates", WINDOW_NORMAL);
    setWindowProperty("Dense Optical Flow with Angular Rates", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    
    // For roll integration (if desired for an indicator, not used in LS)
    double prevTime = (double)getTickCount() / getTickFrequency();
    
    while (true) {
        cap >> frame;
        if (frame.empty())
            break;
        
        // Convert frame to grayscale and downscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        resize(gray, smallGray, Size(), scaleFactor, scaleFactor);
        
        // Compute dense optical flow on the downscaled images using Farneback's algorithm
        Mat flow;
        calcOpticalFlowFarneback(smallPrevGray, smallGray, flow,
                                 0.5,   // pyramid scale
                                 3,     // levels
                                 15,    // window size
                                 3,     // iterations
                                 5,     // poly_n
                                 1.2,   // poly_sigma
                                 0);    // flags
        
        // --- Advanced Approach: Estimate Angular Rates via Least Squares ---
        // Our revised model (with origin at image center):
        //   u = - f_small * ω_y - (y - c_y) * ω_z
        //   v =   f_small * ω_x + (x - c_x) * ω_z
        int step = 10;
        int cx = smallGray.cols / 2;
        int cy = smallGray.rows / 2;
        
        vector<Mat> A_rows;
        vector<double> b_vals;
        
        for (int y = 0; y < flow.rows; y += step) {
            for (int x = 0; x < flow.cols; x += step) {
                Point2f fxy = flow.at<Point2f>(y, x);
                double u_val = fxy.x;
                double v_val = fxy.y;
                double x_centered = x - cx;
                double y_centered = y - cy;
                
                // Equation for u: [ 0, -f_small, -y_centered ] dot [ω_x, ω_y, ω_z]^T = u_val
                Mat row1 = (Mat_<double>(1, 3) << 0, -f_small, -y_centered);
                // Equation for v: [ f_small, 0, x_centered ] dot [ω_x, ω_y, ω_z]^T = v_val
                Mat row2 = (Mat_<double>(1, 3) << f_small, 0, x_centered);
                
                A_rows.push_back(row1);
                A_rows.push_back(row2);
                b_vals.push_back(u_val);
                b_vals.push_back(v_val);
            }
        }
        
        int numRows = A_rows.size();
        Mat A(numRows, 3, CV_64F);
        Mat b(numRows, 1, CV_64F);
        for (int i = 0; i < numRows; i++) {
            A_rows[i].copyTo(A.row(i));
            b.at<double>(i, 0) = b_vals[i];
        }
        
        Mat omega;
        solve(A, b, omega, DECOMP_SVD);
        double omega_x = omega.at<double>(0, 0);  // pitch rate (rad/s)
        double omega_y = omega.at<double>(1, 0);  // yaw rate (rad/s)
        double omega_z = omega.at<double>(2, 0);  // roll rate (rad/s)
        
        // --- Visualization ---
        // Draw flow vectors on a copy of the downscaled gray image (converted to BGR)
        Mat flowVis;
        cvtColor(smallGray, flowVis, COLOR_GRAY2BGR);
        for (int y = 0; y < flow.rows; y += step) {
            for (int x = 0; x < flow.cols; x += step) {
                Point2f fxy = flow.at<Point2f>(y, x);
                line(flowVis, Point(x, y),
                     Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
                     Scalar(0, 255, 0), 1);
                circle(flowVis, Point(x, y), 1, Scalar(0, 0, 255), -1);
            }
        }
        
        // --- Overlay Angular Rate Labels ---
        double fontScale = 0.5;
        int thickness = 1;
        int outline_thickness = 2;
        int baseX = 30;
        int baseY = 30;
        int lineSpacing = 20;
        
        ostringstream oss1, oss2, oss3;
        oss1 << "pitchrate: " << std::showpos << fixed << setprecision(2) << omega_x << " rad/s";
        oss2 << "yaw rate: " << std::showpos << fixed << setprecision(2) << omega_y << " rad/s";
        oss3 << "roll rate: " << std::showpos << fixed << setprecision(2) << omega_z << " rad/s";
        
        // Draw black outline then white text
        Point org1(baseX, baseY);
        Point org2(baseX, baseY + lineSpacing);
        Point org3(baseX, baseY + 2 * lineSpacing);
        
        putText(flowVis, oss1.str(), org1, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,0,0), outline_thickness);
        putText(flowVis, oss1.str(), org1, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255,255,255), thickness);
        putText(flowVis, oss2.str(), org2, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,0,0), outline_thickness);
        putText(flowVis, oss2.str(), org2, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255,255,255), thickness);
        putText(flowVis, oss3.str(), org3, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,0,0), outline_thickness);
        putText(flowVis, oss3.str(), org3, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255,255,255), thickness);
        
        imshow("Dense Optical Flow with Angular Rates", flowVis);
        if (waitKey(1) == 27)  // Exit if ESC is pressed
            break;
        
        // Prepare for next iteration
        smallPrevGray = smallGray.clone();
    }
    
    cap.release();
    destroyAllWindows();
    unlink(fifoPath.c_str());
    return 0;
}







// // Wrap C headers in extern "C" for proper C linkage in C++
// extern "C" {
//     #include <unistd.h>
//     #include <sys/stat.h>
// }

// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <sstream>
// #include <iomanip>
// #include <cstdlib>
// #include <vector>
// #include <cmath>

// using namespace std;
// using namespace cv;

// int main() {
//     // ----- Set up libcamera-vid capture via FIFO -----
//     string fifoPath = "/tmp/libcamera_vid_fifo";
//     unlink(fifoPath.c_str());
//     if (mkfifo(fifoPath.c_str(), 0666) < 0) {
//         perror("mkfifo");
//         exit(-1);
//     }
    
//     string libcamCmd = "libcamera-vid --nopreview --mode 2328:1748 --inline -t 0 --output " 
//                          + fifoPath + " &";
//     cout << "Starting libcamera-vid with command:\n" << libcamCmd << endl;
//     system(libcamCmd.c_str());
    
//     string pipeline = "filesrc location=" + fifoPath +
//                       " ! h264parse ! v4l2h264dec ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
//     cout << "Opening VideoCapture with pipeline:\n" << pipeline << endl;
//     VideoCapture cap(pipeline, CAP_GSTREAMER);
//     if (!cap.isOpened()) {
//         cerr << "Error: Could not open VideoCapture with pipeline." << endl;
//         exit(-1);
//     }
    
//     // ----- Grab the first frame and initialize -----
//     Mat frame;
//     cap >> frame;
//     if (frame.empty()) {
//         cerr << "Error: First frame is empty." << endl;
//         return -1;
//     }
    
//     // Downscale factor (adjust as needed)
//     double scaleFactor = 0.5;
    
//     // Convert first frame to grayscale and then downscale
//     Mat gray, smallGray, smallPrevGray;
//     cvtColor(frame, gray, COLOR_BGR2GRAY);
//     resize(gray, smallPrevGray, Size(), scaleFactor, scaleFactor);
    
//     // Compute effective focal length for the downscaled image.
//     // Assume horizontal FOV of 90°.
//     double FOV_H = 90.0; // degrees
//     double smallWidth = frame.cols * scaleFactor;
//     double f_small = (smallWidth / 2.0) / tan((FOV_H * CV_PI / 180.0) / 2.0);
    
//     // Create a full screen window
//     namedWindow("Dense Optical Flow with Angular Rates", WINDOW_NORMAL);
//     setWindowProperty("Dense Optical Flow with Angular Rates", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    
//     while (true) {
//         cap >> frame;
//         if (frame.empty())
//             break;
        
//         // Convert frame to grayscale and downscale
//         cvtColor(frame, gray, COLOR_BGR2GRAY);
//         resize(gray, smallGray, Size(), scaleFactor, scaleFactor);
        
//         // Compute dense optical flow on the downscaled images using Farneback's algorithm
//         Mat flow;
//         calcOpticalFlowFarneback(smallPrevGray, smallGray, flow,
//                                  0.5,   // pyramid scale
//                                  3,     // levels
//                                  15,    // window size
//                                  3,     // iterations
//                                  5,     // poly_n
//                                  1.2,   // poly_sigma
//                                  0);    // flags
        
//         // --- Advanced Approach: Estimate Angular Rates via Least Squares ---
//         // For each pixel (in the downscaled image, with origin at the center):
//         //   u = ω_z*(x - cx) - ω_y*f_small
//         //   v = ω_z*(y - cy) + ω_x*f_small
//         int step = 10;
//         int cx = smallGray.cols / 2;
//         int cy = smallGray.rows / 2;
        
//         vector<Mat> A_rows;
//         vector<double> b_vals;
        
//         for (int y = 0; y < flow.rows; y += step) {
//             for (int x = 0; x < flow.cols; x += step) {
//                 Point2f fxy = flow.at<Point2f>(y, x);
//                 double u_val = fxy.x;
//                 double v_val = fxy.y;
//                 double x_centered = x - cx;
//                 double y_centered = y - cy;
                
//                 // Equation for u: [0, -f_small, x_centered] * [ω_x, ω_y, ω_z]^T = u_val
//                 Mat row1 = (Mat_<double>(1, 3) << 0, -f_small, x_centered);
//                 // Equation for v: [f_small, 0, y_centered] * [ω_x, ω_y, ω_z]^T = v_val
//                 Mat row2 = (Mat_<double>(1, 3) << f_small, 0, y_centered);
                
//                 A_rows.push_back(row1);
//                 A_rows.push_back(row2);
//                 b_vals.push_back(u_val);
//                 b_vals.push_back(v_val);
//             }
//         }
        
//         int numRows = A_rows.size();
//         Mat A(numRows, 3, CV_64F);
//         Mat b(numRows, 1, CV_64F);
//         for (int i = 0; i < numRows; i++) {
//             A_rows[i].copyTo(A.row(i));
//             b.at<double>(i, 0) = b_vals[i];
//         }
        
//         Mat omega;
//         solve(A, b, omega, DECOMP_SVD);
//         double omega_x = omega.at<double>(0, 0);  // pitch rate (rad/s)
//         double omega_y = omega.at<double>(1, 0);  // yaw rate (rad/s)
//         double omega_z = omega.at<double>(2, 0);  // roll rate (rad/s)
        
//         // --- Visualization ---
//         // Draw flow vectors on a copy of the downscaled gray image (converted to BGR)
//         Mat flowVis;
//         cvtColor(smallGray, flowVis, COLOR_GRAY2BGR);
//         for (int y = 0; y < flow.rows; y += step) {
//             for (int x = 0; x < flow.cols; x += step) {
//                 Point2f fxy = flow.at<Point2f>(y, x);
//                 line(flowVis, Point(x, y),
//                      Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
//                      Scalar(0, 255, 0), 1);
//                 circle(flowVis, Point(x, y), 1, Scalar(0, 0, 255), -1);
//             }
//         }
        
//         // --- Overlay Angular Rate Labels ---
//         double fontScale = 0.5;
//         int thickness = 1;
//         int outline_thickness = 2;
//         int baseX = 30;
//         int baseY = 30;
//         int lineSpacing = 20;
        
//         ostringstream oss1, oss2, oss3;
//         oss1 << std::showpos << "pitchrate: " << fixed << setprecision(2) << omega_x << " rad/s";
//         oss2 << std::showpos << "yaw rate: " << fixed << setprecision(2) << omega_y << " rad/s";
//         oss3 << std::showpos << "roll rate: " << fixed << setprecision(2) << omega_z << " rad/s";
        
//         // Draw black outline, then white text for each line
//         Point org1(baseX, baseY);
//         Point org2(baseX, baseY + lineSpacing);
//         Point org3(baseX, baseY + 2 * lineSpacing);
        
//         // Outline for pitchrate
//         putText(flowVis, oss1.str(), org1, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,0,0), outline_thickness);
//         putText(flowVis, oss1.str(), org1, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255,255,255), thickness);
//         // Outline for yaw rate
//         putText(flowVis, oss2.str(), org2, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,0,0), outline_thickness);
//         putText(flowVis, oss2.str(), org2, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255,255,255), thickness);
//         // Outline for roll rate
//         putText(flowVis, oss3.str(), org3, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0,0,0), outline_thickness);
//         putText(flowVis, oss3.str(), org3, FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255,255,255), thickness);
        
//         imshow("Dense Optical Flow with Angular Rates", flowVis);
//         if (waitKey(1) == 27)  // Exit if ESC is pressed
//             break;
        
//         // Prepare for next iteration
//         smallPrevGray = smallGray.clone();
//     }
    
//     cap.release();
//     destroyAllWindows();
//     unlink(fifoPath.c_str());
//     return 0;
// }


