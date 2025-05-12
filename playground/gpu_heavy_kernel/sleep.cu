#include <stdio.h>
#include <cuda_runtime.h>


static constexpr int kDurns = 500; 

__global__ void sleep_kernel() {
    __nanosleep(kDurns);
}


int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


    int numSMs = prop.multiProcessorCount;
    int threadsPerBlock = 1024;  // Max threads per block
    int numBlocks;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, sleep_kernel, 1024, 0);
    printf("numBlocks %d\n", numBlocks);
    int blocksPerGrid = numSMs * numBlocks;


    printf("Launching sleep kernel on H100...\n");
    printf("Device: %s\n", prop.name);
    printf("SMs: %d, launching %d blocks\n", numSMs, blocksPerGrid);

    // Kernel launch
    /* for (int i = 0; i < 10000000; i++) { */
    while (true) {
        sleep_kernel<<<blocksPerGrid, threadsPerBlock>>>();
    }
    cudaDeviceSynchronize();

    return 0;
}
