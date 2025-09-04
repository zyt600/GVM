#include <dlfcn.h>
#include <string>

#include "cuda_func_caller.hpp"
#include "logging.hpp"

namespace gvm {

// TODO: (yifan) intercept dlsym too. Need to do it with dlvsym.

CudaFuncCaller::CudaFuncCaller() {
  initializeFunctionRegistry();
  initialize();
}

CudaFuncCaller &CudaFuncCaller::getInstance() {
  static CudaFuncCaller instance_;
  return instance_;
}

CudaFuncCaller::~CudaFuncCaller() {
  if (cuda_lib_handle_) {
    dlclose(cuda_lib_handle_);
    cuda_lib_handle_ = nullptr;
  }
}

bool CudaFuncCaller::initialize() {
  if (initialized_) {
    return true;
  }

  GVM_LOG_INFO("Initializing string-based CUDA function caller...");

  if (!loadCudaLibrary()) {
    GVM_LOG_ERROR("Failed to load CUDA library");
    return false;
  }

  preloadMarkedFunctions();
  initialized_ = true;

  // Count loaded functions
  int preloaded_count = 0;
  for (const auto &entry : function_table_) {
    if (entry.second.cached_ptr != nullptr) {
      preloaded_count++;
    }
  }

  GVM_LOG_INFO_S << "Successfully initialized with " << preloaded_count << "/"
                 << function_table_.size() << " functions pre-loaded";
  return true;
}

bool CudaFuncCaller::loadCudaLibrary() {
  if (cuda_lib_handle_ != nullptr) {
    return true;
  }

  // Try different possible CUDA runtime library names
  const char *cuda_lib_names[] = {"libcudart.so.12", // CUDA 12.x
                                  "libcudart.so.11", // CUDA 11.x
                                  "libcudart.so",    // Generic
                                  nullptr};

  // First try RTLD_NOLOAD to find already loaded library
  for (int i = 0; cuda_lib_names[i] != nullptr; i++) {
    cuda_lib_handle_ = dlopen(cuda_lib_names[i], RTLD_LAZY | RTLD_NOLOAD);
    if (cuda_lib_handle_ != nullptr) {
      GVM_LOG_INFO_S << "Found already loaded CUDA library: "
                     << cuda_lib_names[i];
      return true;
    }
  }

  // If RTLD_NOLOAD failed, try loading normally
  for (int i = 0; cuda_lib_names[i] != nullptr; i++) {
    cuda_lib_handle_ = dlopen(cuda_lib_names[i], RTLD_LAZY);
    if (cuda_lib_handle_ != nullptr) {
      GVM_LOG_INFO_S << "Loaded CUDA library: " << cuda_lib_names[i];
      return true;
    }
  }

  GVM_LOG_ERROR("Failed to load any CUDA library");
  return false;
}

void CudaFuncCaller::initializeFunctionRegistry() {
  // Register commonly used functions for pre-loading
  registerFunction("cudaMalloc", true);
  registerFunction("cudaMallocManaged", true);
  registerFunction("cudaMallocAsync", true);
  registerFunction("cudaFree", true);
  registerFunction("cudaMemGetInfo", true);
  registerFunction("cudaMemAdvise", true);
  registerFunction("cudaMemPrefetchAsync", true);
  registerFunction("cudaGetDevice", true);
  registerFunction("cudaSetDevice", true);
  registerFunction("cudaDeviceSynchronize", true);
  registerFunction("cudaStreamCreate", true);
  registerFunction("cudaStreamDestroy", true);
  registerFunction("cudaMemcpy", true);
  registerFunction("cudaMemcpyAsync", true);

  // Register less common functions for on-demand loading
  registerFunction("cudaMemcpy2D", false);
  registerFunction("cudaMemcpy2DAsync", false);
  registerFunction("cudaMemset", false);
  registerFunction("cudaMemsetAsync", false);
  registerFunction("cudaEventCreate", false);
  registerFunction("cudaEventDestroy", false);
  registerFunction("cudaEventRecord", false);
  registerFunction("cudaEventSynchronize", false);
  registerFunction("cudaEventElapsedTime", false);
}

void CudaFuncCaller::registerFunction(const std::string &name, bool preload) {
  function_table_[name] = FunctionEntry(preload);
}

void CudaFuncCaller::preloadMarkedFunctions() {
  GVM_LOG_INFO("Pre-loading functions marked for pre-loading...");

  int loaded_count = 0;
  int attempted_count = 0;

  for (auto &entry : function_table_) {
    if (entry.second.preload) {
      attempted_count++;
      void *func_ptr = dlsym(cuda_lib_handle_, entry.first.c_str());
      if (func_ptr != nullptr) {
        entry.second.cached_ptr = func_ptr;
        loaded_count++;
      } else {
        GVM_LOG_WARN_S << "Warning: Could not pre-load " << entry.first << " - "
                       << dlerror();
      }
    }
  }

  GVM_LOG_INFO_S << "Pre-loaded " << loaded_count << "/" << attempted_count
                 << " marked functions";
}

void *CudaFuncCaller::getFunction(const std::string &name) {
  if (!initialized_) {
    GVM_LOG_ERROR("Not initialized, call initialize() first");
    return nullptr;
  }

  // Check if function is registered in our table
  auto it = function_table_.find(name);
  if (it != function_table_.end()) {
    // Check if already cached
    if (it->second.cached_ptr != nullptr) {
      return it->second.cached_ptr;
    }
    // Load on demand for registered function
    return loadFunctionOnDemand(name);
  }

  // Function not registered - load and cache directly
  void *func_ptr = dlsym(cuda_lib_handle_, name.c_str());
  if (func_ptr == nullptr) {
    GVM_LOG_ERROR_S << "Failed to load function " << name << " - " << dlerror();
    return nullptr;
  }

  // Register and cache the new function
  function_table_[name] = FunctionEntry(false);
  function_table_[name].cached_ptr = func_ptr;
  GVM_LOG_INFO_S << "Loaded and registered new function: " << name;
  return func_ptr;
}

void *CudaFuncCaller::loadFunctionOnDemand(const std::string &name) {
  GVM_LOG_INFO_S << "Loading function on-demand: " << name;

  void *func_ptr = dlsym(cuda_lib_handle_, name.c_str());
  if (func_ptr == nullptr) {
    GVM_LOG_ERROR_S << "Failed to load function " << name << " - " << dlerror();
    return nullptr;
  }

  // Cache for future use
  function_table_[name].cached_ptr = func_ptr;
  return func_ptr;
}

} // namespace gvm
