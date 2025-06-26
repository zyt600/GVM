#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <atomic>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
// #include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include "cuda_function_types.h"
#include "cuda_utils.hpp"

namespace {

// Constants
static constexpr const char *CUDART_LIBRARY_PREFIX = "libcudart.so";
static constexpr bool kCallOriginal = false;

// FIXME: this is not thread safe
static std::unordered_map<void *, size_t> g_cuda_mem_map;

static std::atomic<int64_t> g_cuda_mem_allocated(0);
static int64_t g_cuda_mem_total = 0;

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

  if (kCallOriginal) {
    return CUDA_ENTRY_CALL(Malloc, devPtr, size);
  }

  if (first_call) {
    std::cout << "[INTERCEPTOR] called cudaMalloc! Replacing with "
                 "cudaMallocManaged for size: "
              << size << std::endl;
    first_call = false;
  }

  if (g_cuda_mem_total == 0) {
    size_t _cuda_mem_total = 0;
    CUDA_ENTRY_CALL(MemGetInfo, nullptr, &_cuda_mem_total);
    g_cuda_mem_total = _cuda_mem_total;
  }
  if (g_cuda_mem_allocated + size > g_cuda_mem_total) {
    std::cerr << "[INTERCEPTOR] cudaMalloc: out of memory." << std::endl;
    return cudaErrorMemoryAllocation;
  }

  cudaError_t ret =
      CUDA_ENTRY_CALL(MallocManaged, devPtr, size, cudaMemAttachGlobal);
  if (ret != cudaSuccess) {
    std::cerr << "[INTERCEPTOR] cudaMallocManaged: out of memory." << std::endl;
    return ret;
  }

  g_cuda_mem_map[*devPtr] = size;
  g_cuda_mem_allocated += size;
  std::cout << "total cuda memory allocated: "
            << g_cuda_mem_allocated / 1024 / 1024 << "MB" << std::endl;

  return ret;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
  (void)stream; // suppress warning about unused stream

  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaMallocAsync." << std::endl;
  }

  cudaError_t ret =
      CUDA_ENTRY_CALL(MallocManaged, devPtr, size, cudaMemAttachGlobal);
  if (ret != cudaSuccess) {
    std::cerr << "[INTERCEPTOR] cudaMallocAsync: out of memory." << std::endl;
    return ret;
  }

  g_cuda_mem_map[*devPtr] = size;
  g_cuda_mem_allocated += size;
  std::cout << "total cuda memory allocated: "
            << g_cuda_mem_allocated / 1024 / 1024 << "MB" << std::endl;

  return ret;
}

cudaError_t cudaFree(void *devPtr) {
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaFree." << std::endl;
  }

  auto it = g_cuda_mem_map.find(devPtr);
  if (it != g_cuda_mem_map.end()) {
    size_t size = it->second;
    g_cuda_mem_map.erase(it);
    g_cuda_mem_allocated -= size;
  }

  return CUDA_ENTRY_CALL(Free, devPtr);
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    std::cout << "[INTERCEPTOR] called cudaMemGetInfo." << std::endl;
  }

  cudaError_t ret = CUDA_ENTRY_CALL(MemGetInfo, free, total);
  if (ret != cudaSuccess) {
    return ret;
  }

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