cmake_minimum_required(VERSION 3.10)
project(simple_pose)

find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)
find_package(Threads REQUIRED)

add_executable(${PROJECT_NAME} simple_pose.cpp)
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort ${OpenCV_LIBS} Threads::Threads)
