// cuda_driver_api_example.cpp

#include <cuda.h>
#include <iostream>
#include <fstream>

#define N 16

#define CHECK_CUDA(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char *errStr; \
            cuGetErrorString(err, &errStr); \
            std::cerr << "CUDA Error: " << errStr << " at line " << __LINE__ << "\n"; \
            exit(1); \
        } \
    } while (0)

int main() {
    CHECK_CUDA(cuInit(0));

    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    int h_data[N];
    for (int i = 0; i < N; ++i) h_data[i] = i;

    CUdeviceptr d_data;
    CHECK_CUDA(cuMemAlloc(&d_data, N * sizeof(int)));
    CHECK_CUDA(cuMemcpyHtoD(d_data, h_data, N * sizeof(int)));

    CUmodule module;
    CHECK_CUDA(cuModuleLoad(&module, "kernel.ptx"));  // Load compiled PTX
    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "simple_add"));

    void *args[] = { &d_data };

    CHECK_CUDA(cuLaunchKernel(kernel,
                              1, 1, 1,
                              N, 1, 1,
                              0, 0,
                              args, 0));

    CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuMemcpyDtoH(h_data, d_data, N * sizeof(int)));

    for (int i = 0; i < N; ++i)
        std::cout << h_data[i] << " ";
    std::cout << "\n";

    cuMemFree(d_data);
    cuCtxDestroy(context);
    return 0;
}
