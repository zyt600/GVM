cmake_minimum_required(VERSION 3.5)
project(scheduler C CXX)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB REQUIRED glib-2.0)

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