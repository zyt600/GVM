#include <cuda_runtime.h>
#include <cstdio>

__global__ void hog_h100_kernel() {
    // Max shared memory per block (228 KB)
    __shared__ float smem[228 * 1024 / sizeof(float)];

    // Use ~64 registers per thread (approximate, depends on compiler)
    float r[64];
    for (int i = 0; i < 64; ++i) r[i] = threadIdx.x + i;

    // Infinite loop to persist
    while (true) {
        // Do some dummy computations
        for (int i = 1; i < 64; ++i) {
            r[0] += r[i] * 0.00001f;
        }

        // Write to shared memory to ensure usage
        smem[threadIdx.x % (sizeof(smem)/sizeof(float))] = r[0];

        __syncthreads(); // Force memory sync
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


    int numSMs = prop.multiProcessorCount;
    int threadsPerBlock = 1024;  // Max threads per block
    int blocksPerGrid = numSMs;  // 1 block per SM for full persistence

    int numBlocks;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, hog_h100_kernel, 1024, 0);
    printf("numBlocks %d\n", numBlocks);
    numBlocks = 1;
    blocksPerGrid *= numBlocks;

    printf("Launching persistent kernel on H100...\n");
    printf("Device: %s\n", prop.name);
    printf("SMs: %d, launching %d blocks\n", numSMs, blocksPerGrid);

    // Kernel launch
    hog_h100_kernel<<<blocksPerGrid, threadsPerBlock>>>();

    cudaDeviceSynchronize(); // Will never return
    return 0;
}
