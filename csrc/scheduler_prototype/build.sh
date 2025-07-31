#!/bin/bash

# Default options
CLEAN_BUILD=false
BUILD_TYPE="Release"
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            HELP=true
            shift
            ;;
    esac
done

# Show help
if [ "$HELP" = true ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --clean         Delete build directory (full clean build)"
    echo "  --debug         Build in Debug mode (default: Release)"
    echo "  --release       Build in Release mode (default)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Incremental build (default)"
    echo "  $0 --clean     # Full clean build"
    echo "  $0 --debug     # Incremental debug build"
    echo "  $0 --clean --debug # Clean debug build"
    exit 0
fi

function build() {
    set -o errexit
    set -o pipefail
    set -o nounset
    set -o xtrace

    # Unset LD_PRELOAD during build to avoid errors
    unset LD_PRELOAD

    ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)

    # Conditionally clean build directory
    if [ "$CLEAN_BUILD" = true ]; then
        echo "Performing clean build..."
        rm -rf ${ROOT}/build
        mkdir ${ROOT}/build
    else
        echo "Performing incremental build..."
        mkdir -p ${ROOT}/build
    fi

    cd ${ROOT}/build
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
    make -j

    # Copy libraries to scheduler-lib
    rm -rf ${ROOT}/scheduler-lib
    mkdir ${ROOT}/scheduler-lib
    cd ${ROOT}/scheduler-lib
    cp ${ROOT}/build/libcuda-control-scheduler.so ./libcuda-control.so

    touch ./ld.so.preload
    echo -e "/libcontroller.so\n/libcuda.so\n/libcuda.so.1\n/libnvidia-ml.so\n/libnvidia-ml.so.1" > ./ld.so.preload
    cp libcuda-control.so ./libnvidia-ml.so.1
    patchelf --set-soname libnvidia-ml.so.1 ./libnvidia-ml.so.1
    cp libcuda-control.so ./libnvidia-ml.so
    patchelf --set-soname libnvidia-ml.so ./libnvidia-ml.so
    cp libcuda-control.so ./libcuda.so.1
    patchelf --set-soname libcuda.so.1 ./libcuda.so.1
    cp libcuda-control.so ./libcuda.so
    patchelf --set-soname libcuda.so ./libcuda.so
    cp libcuda-control.so ./libcontroller.so
    patchelf --set-soname libcontroller.so ./libcontroller.so

    cd ..
}

build
