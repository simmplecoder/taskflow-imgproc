cmake_minimum_required(VERSION 3.18)
project(taskflow-imgproc LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(TIFF REQUIRED)

add_library(gil INTERFACE)
target_include_directories(gil INTERFACE ${JPEG_INCLUDE_DIR})
target_link_libraries(gil INTERFACE ${JPEG_LIBRARIES} TIFF::TIFF PNG::PNG)

find_package(Taskflow REQUIRED)
find_package(blazing-gil REQUIRED)

add_library(blas_lib INTERFACE)
find_package(BLAS REQUIRED)
target_link_libraries(blas_lib INTERFACE BLAS::BLAS)
target_include_directories(blas_lib INTERFACE $ENV{MKLROOT}/include)

find_package(spdlog REQUIRED)

add_executable(prog main.cpp)
target_link_libraries(prog PRIVATE Boost::headers Taskflow::Taskflow blazing-gil::blazing-gil gil blas_lib)

add_executable(cuda-convolve convolve.cu)
set_property(TARGET cuda-convolve PROPERTY CUDA_ARCHITECTURES 61)
set_property(TARGET cuda-convolve PROPERTY CUDA_STANDARD 14)
target_link_libraries(cuda-convolve PRIVATE Boost::headers blazing-gil::blazing-gil gil blas_lib spdlog::spdlog)