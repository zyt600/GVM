#include <iostream>
#include <cuda_runtime.h>

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            std::cerr << "Error code: " << err << std::endl; \
            return 1; \
        } \
    } while(0)

__global__ void fill_kernel(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

int main() {
    const int N = 16;
    int *d_data, *h_data;

    std::cout << "Starting CUDA operations..." << std::endl;

    // Print CUDA device information
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    if (deviceCount > 0) {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
        std::cout << "Device name: " << deviceProp.name << std::endl;
        std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    }

    // Allocate host memory
    h_data = new int[N];
    if (!h_data) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return 1;
    }

    // Allocate device memory
    std::cout << "Attempting to allocate device memory..." << std::endl;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    std::cout << "Device memory allocated successfully" << std::endl;

    // Launch kernel with 1 block of 16 threads
    std::cout << "Launching kernel in application..." << std::endl;
    fill_kernel<<<1, N>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    std::cout << "Kernel launched successfully" << std::endl;

    // Wait for GPU to finish
    std::cout << "Synchronizing device..." << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Device synchronized successfully" << std::endl;

    // Copy result back to host
    std::cout << "Copying memory back to host..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Memory copied back to host successfully" << std::endl;

    // Print results
    std::cout << "Results: ";
    for (int i = 0; i < N; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;

    std::cout << "CUDA operations completed successfully" << std::endl;
    return 0;
}
