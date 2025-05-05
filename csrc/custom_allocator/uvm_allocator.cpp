// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/cumem_allocator.cpp A
// CUDAPluggableAllocator based on UVM APIs.
#include <array>
#include <cstddef>
#include <iostream>

extern "C" {

#include <cuda.h>
#include <sys/types.h>

// Dummy pointers for zero-size allocations
static constexpr size_t MAX_NUM_GPUS = 72;
static std::array<void *, MAX_NUM_GPUS> dummy_ptr{};

char error_msg[10240]; // 10KB buffer to store error messages
CUresult no_error = CUresult(0);
CUresult error_code = no_error; // store error code

#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    CUresult error = condition;                                                \
    if (error != 0) {                                                          \
      error_code = error;                                                      \
      char *error_string;                                                      \
      cuGetErrorString(error, (const char **)&error_string);                   \
      snprintf(error_msg, sizeof(error_msg), "CUDA Error: %s at %s:%d",        \
               error_string, __FILE__, __LINE__);                              \
      std::cerr << error_msg << std::endl;                                     \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Helper functions:

void ensure_context(unsigned long long device) {
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // Ensure device context.
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
}

// ---------------------------------------------------------------------------
// Our exported C functions for memory allocation and deallocation

// use CUstream instead of cudaStream_t, to avoid including cuda_runtime_api.h
void *uvm_alloc(ssize_t size, int device, CUstream stream) {
  ensure_context(device);

  // first allocation, align the size, and reserve an address, and also
  // allocate a CUmemGenericAllocationHandle

  // // Align the size to the allocation granularity
  // CUmemAllocationProp prop = {};
  // prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  // prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // prop.location.id = device;
  // prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

  // // Check if the allocation is supported
  // size_t granularity;
  // CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
  //                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  // if (error_code != 0) {
  //   return nullptr;
  // }
  // size_t alignedSize = ((size + granularity - 1) / granularity) *
  // granularity; size = alignedSize;

  if (size != 0) {
    CUdeviceptr d_mem;
    CUDA_CHECK(cuMemAllocManaged(&d_mem, size, CU_MEM_ATTACH_GLOBAL));
    if (error_code != 0) {
      return nullptr;
    }
    // CUDA_CHECK(
    //     cuMemAdvise(d_mem, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, 0));
    // if (error_code != 0) {
    //   CUDA_CHECK(cuMemFree(d_mem));
    //   return nullptr;
    // }

    return reinterpret_cast<void *>(d_mem);
  }

  // for zero-size allocation, we use a dummy pointer per device
  if (dummy_ptr[device] == nullptr) { // lazy init
    CUDA_CHECK(
        cuMemAllocManaged(reinterpret_cast<CUdeviceptr *>(&dummy_ptr[device]),
                          sizeof(void *), CU_MEM_ATTACH_GLOBAL));
    if (error_code != 0) {
      return nullptr;
    }
  }
  return dummy_ptr[device];
}

// use CUstream instead of cudaStream_t, to avoid including cuda_runtime_api.h
void uvm_free(void *ptr, ssize_t size, int device, CUstream stream) {
  if (size == 0)
    return;

  CUdeviceptr d_mem = reinterpret_cast<CUdeviceptr>(ptr);
  CUDA_CHECK(cuMemFree(d_mem));
}
} // extern "C"