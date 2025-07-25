cmake_minimum_required(VERSION 3.18)

project(custom_cuda_lib VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Don't find CUDA package to avoid automatic linking
# find_package(CUDAToolkit REQUIRED)

set(SOURCES
    custom_cuda.cpp
    memory_manager.cpp
    cuda_func_caller.cpp
)

set(HEADERS
    cuda_utils.hpp
    memory_manager.h
    cuda_func_caller.h
)

# Create shared library for LD_PRELOAD usage
add_library(custom_cuda SHARED ${SOURCES})

target_link_libraries(custom_cuda 
    PRIVATE 
    gvmdrv
    ${CMAKE_DL_LIBS}
)

target_include_directories(custom_cuda
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(custom_cuda PROPERTIES
    OUTPUT_NAME "custom_cuda"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
    PREFIX "lib"
)

target_compile_options(custom_cuda PRIVATE
    -fPIC
    -shared
)

# Add linker flags for LD_PRELOAD compatibility
target_link_options(custom_cuda PRIVATE
    -shared
    -fPIC
)

install(TARGETS custom_cuda
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS}
    DESTINATION include/custom_cuda_lib
)