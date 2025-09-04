cmake_minimum_required(VERSION 3.18)

project(
  libgvmdrv
  VERSION 1.0.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(SOURCES gvmdrv.cpp gvmdrv_log.cpp)

set(HEADERS gvmdrv.h gvmdrv_ioctl.h gvmdrv_log.h)

if(BUILD_SHARED_LIBS)
  add_library(gvmdrv SHARED ${SOURCES})
  set_target_properties(
    gvmdrv
    PROPERTIES OUTPUT_NAME "gvmdrv"
               VERSION ${PROJECT_VERSION}
               SOVERSION 1
               POSITION_INDEPENDENT_CODE ON)
else()
  add_library(gvmdrv STATIC ${SOURCES})
endif()

target_compile_options(gvmdrv PRIVATE -Wall -Wextra -O2 -fPIC)

target_compile_definitions(gvmdrv PRIVATE $<$<CONFIG:Debug>:DEBUG>)

target_include_directories(gvmdrv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_TESTS)
  find_package(CUDAToolkit REQUIRED)

  set(TEST_SOURCES test/test.cpp test/test_basic.cpp test/test_advanced.cpp)

  foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    set(test_target "gvmdrv_${test_name}")
    add_executable(${test_target} ${test_source})
    target_link_libraries(${test_target} PRIVATE gvmdrv CUDA::cudart)
    add_test(NAME ${test_target} COMMAND ${test_target})
  endforeach()
endif()

install(
  TARGETS gvmdrv
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(FILES ${HEADERS} DESTINATION include/libgvmdrv)
