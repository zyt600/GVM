#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

CUDA_BIN=${CUDA_BIN:-/usr/local/cuda/bin}
GVM_INSTALL=${GVM_INSTALL:-$SCRIPT_DIR/gvm-cuda-driver/install}

echo "=== Building test_eviction_notice ==="
$CUDA_BIN/nvcc -o test_eviction_notice test_eviction_notice.cu \
    -lpthread \
    -L"$GVM_INSTALL" -lcuda \
    -lgvmnotify \
    -Xlinker -rpath -Xlinker "$GVM_INSTALL" \
    -Wno-deprecated-gpu-targets

echo "Build OK: $SCRIPT_DIR/test_eviction_notice"
echo "Run:  sudo LD_LIBRARY_PATH=$GVM_INSTALL:\$LD_LIBRARY_PATH ./test_eviction_notice"
