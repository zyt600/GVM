#pragma once

#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace {
// Function type definitions for common CUDA functions
using CudaMallocFunc = cudaError_t (*)(void **, size_t);
using CudaMallocManagedFunc = cudaError_t (*)(void **, size_t, unsigned int);
using CudaMallocAsyncFunc = cudaError_t (*)(void **, size_t, cudaStream_t);
using CudaFreeFunc = cudaError_t (*)(void *);
using CudaMemGetInfoFunc = cudaError_t (*)(size_t *, size_t *);
using CudaMemAdviseFunc = cudaError_t (*)(void *, size_t, cudaMemoryAdvise,
                                          int);
using CudaMemPrefetchAsyncFunc = cudaError_t (*)(const void *, size_t, int,
                                                 cudaStream_t);
using CudaGetDeviceFunc = cudaError_t (*)(int *);
using CudaSetDeviceFunc = cudaError_t (*)(int);
using CudaDeviceSynchronizeFunc = cudaError_t (*)();
using CudaStreamCreateFunc = cudaError_t (*)(cudaStream_t *);
using CudaStreamDestroyFunc = cudaError_t (*)(cudaStream_t);
using CudaMemcpyFunc = cudaError_t (*)(void *, const void *, size_t,
                                       cudaMemcpyKind);
using CudaMemcpyAsyncFunc = cudaError_t (*)(void *, const void *, size_t,
                                            cudaMemcpyKind, cudaStream_t);

// Additional CUDA function types
using CudaMemcpy2DFunc = cudaError_t (*)(void *, size_t, const void *, size_t,
                                         size_t, size_t, cudaMemcpyKind);
using CudaMemcpy2DAsyncFunc = cudaError_t (*)(void *, size_t, const void *,
                                              size_t, size_t, size_t,
                                              cudaMemcpyKind, cudaStream_t);
using CudaMemsetFunc = cudaError_t (*)(void *, int, size_t);
using CudaMemsetAsyncFunc = cudaError_t (*)(void *, int, size_t, cudaStream_t);
using CudaEventCreateFunc = cudaError_t (*)(cudaEvent_t *);
using CudaEventDestroyFunc = cudaError_t (*)(cudaEvent_t);
using CudaEventRecordFunc = cudaError_t (*)(cudaEvent_t, cudaStream_t);
using CudaEventSynchronizeFunc = cudaError_t (*)(cudaEvent_t);
using CudaEventElapsedTimeFunc = cudaError_t (*)(float *, cudaEvent_t,
                                                 cudaEvent_t);

} // namespace