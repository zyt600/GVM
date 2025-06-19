#pragma once

#include <cuda_runtime.h>
#include <iostream>

namespace gvm {

#define CUDA_RT_CHECK(call)                                                    \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return err;                                                              \
    }                                                                          \
  } while (0)
} // namespace gvm
