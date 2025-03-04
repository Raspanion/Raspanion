cmake_minimum_required(VERSION 3.10.0)
project(visual_alt_hold_example)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)

set(COMPILE_OPTIONS_CPP -Wall -Werror -O3)

add_executable(${PROJECT_NAME} visual_alt_hold.cpp)
add_compile_options(-Wno-address-of-packed-member)


# Include dirs
include_directories(/usr/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/home/erikedwards/Repos/c_library_v2)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Werror -O3 -Wno-error=address-of-packed-member)
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort Threads::Threads)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
