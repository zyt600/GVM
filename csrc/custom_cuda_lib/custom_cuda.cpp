#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
// #include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include "cuda_function_types.h"
#include "cuda_utils.hpp"
#include "memory_manager.h"

namespace {

// Constants
static constexpr const char *CUDART_LIBRARY_PREFIX = "libcudart.so";
static constexpr bool kCallOriginal = false;

// Global variables
static void *cuda_lib_handle = nullptr;

// Simple function cache for extensibility
static std::unordered_map<std::string, void *> g_function_cache;
static bool g_symbols_loaded = false;

// Generic function loader that works with any function type
template <typename FuncType> FuncType GetCudaFunction(const std::string &name) {
  auto it = g_function_cache.find(name);
  if (it != g_function_cache.end()) {
    return reinterpret_cast<FuncType>(it->second);
  }

  void *func_ptr = dlsym(cuda_lib_handle, name.c_str());
  if (func_ptr == nullptr) {
    std::cerr << "Failed to load function: " << name << " - " << dlerror()
              << std::endl;
    return nullptr;
  }

  g_function_cache[name] = func_ptr;
  return reinterpret_cast<FuncType>(func_ptr);
}

#define CUDA_FUNC(func_name)                                                   \
  inline Cuda##func_name##Func get_##func_name() {                             \
    return GetCudaFunction<Cuda##func_name##Func>("cuda" #func_name);          \
  }

CUDA_FUNC(Malloc)
CUDA_FUNC(MallocManaged)
CUDA_FUNC(MallocAsync)
CUDA_FUNC(Free)
CUDA_FUNC(MemGetInfo)
CUDA_FUNC(MemAdvise)
CUDA_FUNC(MemPrefetchAsync)
CUDA_FUNC(GetDevice)
CUDA_FUNC(SetDevice)
CUDA_FUNC(DeviceSynchronize)
CUDA_FUNC(StreamCreate)
CUDA_FUNC(StreamDestroy)
CUDA_FUNC(Memcpy)
CUDA_FUNC(MemcpyAsync)

// Helper function to load CUDA symbols - now extensible
bool load_cuda_symbols() {
  if (g_symbols_loaded) {
    return true;
  }

  if (cuda_lib_handle == nullptr) {
    cuda_lib_handle = dlopen(CUDART_LIBRARY_PREFIX, RTLD_LAZY);
    if (cuda_lib_handle == nullptr) {
      std::cerr << "Failed to open CUDA library: " << dlerror() << std::endl;
      return false;
    }
  }

  // Define common CUDA functions to pre-load
  const std::vector<std::string> common_functions = {
      "cudaMalloc",           "cudaMallocManaged",
      "cudaMallocAsync",      "cudaFree",
      "cudaMemGetInfo",       "cudaMemAdvise",
      "cudaMemPrefetchAsync", "cudaGetDevice",
      "cudaSetDevice",        "cudaDeviceSynchronize",
      "cudaStreamCreate",     "cudaStreamDestroy",
      "cudaMemcpy",           "cudaMemcpyAsync",
      "cudaMemcpy2D",         "cudaMemcpy2DAsync",
      "cudaMemset",           "cudaMemsetAsync",
      "cudaEventCreate",      "cudaEventDestroy",
      "cudaEventRecord",      "cudaEventSynchronize",
      "cudaEventElapsedTime"};

  bool all_loaded = true;
  for (const auto &func_name : common_functions) {
    void *func_ptr = dlsym(cuda_lib_handle, func_name.c_str());
    if (func_ptr == nullptr) {
      std::cerr << "Failed to pre-load function: " << func_name << std::endl;
      all_loaded = false;
    } else {
      g_function_cache[func_name] = func_ptr;
    }
  }

  g_symbols_loaded = all_loaded;
  return all_loaded;
}

// Updated macro that uses the new function-specific getters
#define CUDA_ENTRY_CALL(func_name, ...)                                        \
  ({                                                                           \
    if (!load_cuda_symbols()) {                                                \
      std::cerr << "Failed to load CUDA symbols." << std::endl;                \
      return cudaErrorUnknown;                                                 \
    }                                                                          \
    auto fn = get_##func_name();                                               \
    if (!fn) {                                                                 \
      std::cerr << "Failed to find symbol: " << #func_name << std::endl;       \
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
    std::cout << "[INTERCEPTOR] called cudaMalloc! Replacing with "
                 "cudaMallocManaged for size: "
              << size << std::endl;
    void *tmp_ptr;
    cudaError_t ret =
        CUDA_ENTRY_CALL(MallocManaged, &tmp_ptr, 0, cudaMemAttachGlobal);
    if (ret != cudaSuccess) {
      std::cerr << "[INTERCEPTOR] init cudaMallocManaged: failed." << std::endl;
      return ret;
    }
    ret = CUDA_ENTRY_CALL(Free, tmp_ptr);
    if (ret != cudaSuccess) {
      std::cerr << "[INTERCEPTOR] init cudaFree: failed." << std::endl;
      return ret;
    }
  }

  if (kCallOriginal) {
    return CUDA_ENTRY_CALL(Malloc, devPtr, size);
  }

  // Check if allocation is allowed (handles retry logic)
  MemoryManager &memory_mgr = MemoryManager::getInstance();
  // if (!memory_mgr.canAllocate(size)) {
  //   return cudaErrorMemoryAllocation;
  // }

  // Perform the actual allocation using cudaMallocManaged
  cudaError_t ret =
      CUDA_ENTRY_CALL(MallocManaged, devPtr, size, cudaMemAttachGlobal);
  if (ret != cudaSuccess) {
    std::cerr << "[INTERCEPTOR] cudaMallocManaged: out of memory." << std::endl;
    return ret;
  }

  // Record successful allocation
  memory_mgr.recordAllocation(*devPtr, size);
  return ret;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
  (void)stream; // suppress warning about unused stream

  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaMallocAsync." << std::endl;
  }

  // Check if allocation is allowed (handles retry logic)
  MemoryManager &memory_mgr = MemoryManager::getInstance();
  memory_mgr.init();
  if (!memory_mgr.canAllocate(size)) {
    return cudaErrorMemoryAllocation;
  }

  // Perform the actual allocation using cudaMallocManaged
  cudaError_t ret =
      CUDA_ENTRY_CALL(MallocManaged, devPtr, size, cudaMemAttachGlobal);
  if (ret != cudaSuccess) {
    std::cerr << "[INTERCEPTOR] cudaMallocAsync: out of memory." << std::endl;
    return ret;
  }

  // Record successful allocation
  memory_mgr.recordAllocation(*devPtr, size);
  return ret;
}

cudaError_t cudaFree(void *devPtr) {
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaFree." << std::endl;
  }

  // Record deallocation
  MemoryManager::getInstance().recordDeallocation(devPtr);

  return CUDA_ENTRY_CALL(Free, devPtr);
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaMemGetInfo." << std::endl;
  }

  // Get actual GPU memory info first
  size_t actual_free, actual_total;
  cudaError_t ret = CUDA_ENTRY_CALL(MemGetInfo, &actual_free, &actual_total);
  if (ret != cudaSuccess) {
    return ret;
  }

  // Let MemoryManager handle the logic and return appropriate values
  MemoryManager::getInstance().getMemoryInfo(free, total, actual_free,
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