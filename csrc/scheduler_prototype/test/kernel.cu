// kernel.cu
extern "C" __global__ void simple_add(int *data) {
    int idx = threadIdx.x;
    data[idx] += 1;
}