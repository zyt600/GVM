#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda_runtime.h>

#include "cuda_func_caller.hpp"
#include "cuda_utils.hpp"
#include "logging.hpp"
#include "memory_manager.hpp"

namespace {

// Constants
static constexpr bool kCallOriginal = false;

// Updated macro that uses CudaFuncCaller
#define CUDA_ENTRY_CALL(func_name, ...)                                        \
  ({                                                                           \
    auto &cuda_caller = gvm::CudaFuncCaller::getInstance();                    \
    if (!cuda_caller.isInitialized()) {                                        \
      GVM_LOG_ERROR("CUDA caller not initialized");                            \
      return cudaErrorUnknown;                                                 \
    }                                                                          \
    auto fn = cuda_caller.get##func_name();                                    \
    if (!fn) {                                                                 \
      GVM_LOG_ERROR("Failed to find " #func_name);                             \
      return cudaErrorUnknown;                                                 \
    }                                                                          \
    fn(__VA_ARGS__);                                                           \
  })

} // anonymous namespace

extern "C" {

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  static bool first_call = true;

  // Initialize memory manager on first call
  if (first_call) {
    first_call = false;
    GVM_LOG_DEBUG("cudaMalloc is called.");
  }

  if (kCallOriginal) {
    return CUDA_ENTRY_CALL(Malloc, devPtr, size);
  }

  // Check if allocation is allowed (handles retry logic)
  gvm::MemoryManager &memory_mgr = gvm::MemoryManager::getInstance();
  if (!memory_mgr.canAlloc(size)) {
    return cudaErrorMemoryAllocation;
  }

  // Perform the actual allocation using cudaMallocManaged
  cudaError_t ret =
      CUDA_ENTRY_CALL(MallocManaged, devPtr, size, cudaMemAttachGlobal);
  if (ret != cudaSuccess) {
    GVM_LOG_ERROR("cudaMallocManaged: out of memory.");
    return ret;
  }

  // Record successful allocation
  memory_mgr.recordAlloc(*devPtr, size);
  return ret;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
  (void)stream; // suppress warning about unused stream

  static bool first_call = true;
  if (first_call) {
    first_call = false;
    GVM_LOG_DEBUG("cudaMallocAsync is called.");
  }

  // Check if allocation is allowed (handles retry logic)
  gvm::MemoryManager &memory_mgr = gvm::MemoryManager::getInstance();
  memory_mgr.init();
  if (!memory_mgr.canAlloc(size)) {
    return cudaErrorMemoryAllocation;
  }

  // Perform the actual allocation using cudaMallocManaged
  cudaError_t ret =
      CUDA_ENTRY_CALL(MallocManaged, devPtr, size, cudaMemAttachGlobal);
  if (ret != cudaSuccess) {
    GVM_LOG_ERROR("cudaMallocAsync: out of memory.");
    return ret;
  }

  // Record successful allocation
  memory_mgr.recordAlloc(*devPtr, size);
  return ret;
}

cudaError_t cudaFree(void *devPtr) {
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    GVM_LOG_DEBUG("cudaFree is called.");
  }

  // Record deallocation
  gvm::MemoryManager::getInstance().recordDealloc(devPtr);

  return CUDA_ENTRY_CALL(Free, devPtr);
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    GVM_LOG_DEBUG("cudaMemGetInfo is called.");
  }

  // Get actual GPU memory info first
  size_t actual_free, actual_total;
  cudaError_t ret = CUDA_ENTRY_CALL(MemGetInfo, &actual_free, &actual_total);
  if (ret != cudaSuccess) {
    return ret;
  }

  // Let MemoryManager handle the logic and return appropriate values
  gvm::MemoryManager::getInstance().getMemoryInfo(free, total, actual_free,
                                                  actual_total);

  return ret;
}

} // extern "C"

namespace utils {
size_t cuda_available_mem_size() {
  size_t free, total;
  CUDA_RT_CHECK(cudaMemGetInfo(&free, &total));
  return free;
}
} // namespace utils
