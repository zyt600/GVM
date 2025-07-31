cmake_minimum_required(VERSION 3.5)
project(scheduler C CXX)

# Check for local GLib installation first
set(LOCAL_GLIB_DIR "${CMAKE_SOURCE_DIR}/../3rdparty/glib/install")
set(LOCAL_GLIB_PKGCONFIG_DIR "${LOCAL_GLIB_DIR}/lib/pkgconfig")

find_package(PkgConfig REQUIRED)

# Try to find local GLib first
if(EXISTS "${LOCAL_GLIB_PKGCONFIG_DIR}/glib-2.0.pc")
    message(STATUS "Found local GLib installation at ${LOCAL_GLIB_DIR}")
    # Add local pkg-config path
    set(ENV{PKG_CONFIG_PATH} "${LOCAL_GLIB_PKGCONFIG_DIR}:$ENV{PKG_CONFIG_PATH}")
    # Force pkg-config to look in our local directory
    set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
    list(APPEND CMAKE_PREFIX_PATH "${LOCAL_GLIB_DIR}")
endif()

# Find GLib (either local or system)
pkg_search_module(GLIB REQUIRED glib-2.0)

if(GLIB_FOUND)
    message(STATUS "GLib found: ${GLIB_LIBRARIES}")
    message(STATUS "GLib includes: ${GLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "GLib not found. Please install GLib or build it locally in csrc/3rdparty/glib/")
endif()

add_definitions(-D_GNU_SOURCE)
add_compile_options(-Wall -Wshadow -Werror -Wno-format)

include_directories(${CMAKE_SOURCE_DIR}
                    ${GLIB_INCLUDE_DIRS})

link_directories(${GLIB_LIBRARY_DIRS})

if (${ENABLE_DEBUG})
    add_compile_options(-g -O0)
else ()
    add_compile_options(-g -O2)
endif ()

set(STATIC_C_LIBRARIES -static-libgcc -static-libstdc++)

add_library(cuda-control-scheduler SHARED
        src/scheduler/scheduler.c
        include/scheduler/scheduler.h
        include/cuda-interception/cuda_subset.h
        include/cuda-interception/nvml_subset.h
        include/cuda-interception/cuda_helper.h
        include/cuda-interception/nvml_helper.h
        include/cuda-interception/cuda_originals.h
        include/cuda-interception/dlsym_hook.h
        src/cuda-interception/cuda_originals.c
        src/cuda-interception/nvml_entry.c
        src/cuda-interception/loader.c
        src/cuda-interception/dlsym_hook.c)

add_definitions(${GLIB_CFLAGS_OTHER})
target_link_libraries(cuda-control-scheduler ${STATIC_C_LIBRARIES}
                        ${GLIB_LIBRARIES})

target_compile_options(cuda-control-scheduler PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)