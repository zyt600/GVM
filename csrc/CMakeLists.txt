cmake_minimum_required(VERSION 3.18)

project(GVM_CSRC VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build test programs" ON)

add_subdirectory(libgvmdrv)
add_subdirectory(custom_cuda_lib)

if(BUILD_TESTS)
    enable_testing()
endif()