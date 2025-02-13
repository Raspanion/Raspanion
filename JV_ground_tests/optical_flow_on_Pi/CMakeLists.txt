cmake_minimum_required(VERSION 3.10)
project(optical_flow)

find_package(OpenCV REQUIRED)

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})
