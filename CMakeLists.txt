cmake_minimum_required(VERSION 3.15)
project(taskflow-imgproc)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(TIFF REQUIRED)

add_library(gil INTERFACE)
target_include_directories(gil INTERFACE ${JPEG_INCLUDE_DIR})
target_link_libraries(gil INTERFACE ${JPEG_LIBRARIES} TIFF::TIFF PNG::PNG)

find_package(Taskflow)

add_executable(prog main.cpp)
target_link_libraries(prog Boost::headers Taskflow::Taskflow gil)