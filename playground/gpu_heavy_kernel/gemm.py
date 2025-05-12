import torch
import time

# Config: matrix size (A: MxK, B: KxN)
M, K, N = 8192, 8192, 8192  # Large enough to saturate GPU
dtype = torch.float16  # H100 is optimized for float16 / bfloat16
device = 'cuda'

# Prepare input tensors
A = torch.randn((M, K), device=device, dtype=dtype)
B = torch.randn((K, N), device=device, dtype=dtype)

# Warm-up
for _ in range(10):
    _ = torch.matmul(A, B)

print("Starting benchmark... Press Ctrl+C to stop.")
try:
    count = 0
    total_time = 0.0

    while True:
        torch.cuda.synchronize()
        start = time.time()

        # Perform matmul
        C = torch.matmul(A, B)

        torch.cuda.synchronize()
        end = time.time()

        duration = end - start
        tflops = (2 * M * N * K) / duration / 1e12

        total_time += duration
        count += 1

        print(f"[{count}] Time: {duration:.5f} sec | TFLOPS: {tflops:.2f}")

except KeyboardInterrupt:
    print(f"\nBenchmark stopped after {count} iterations.")
    avg_time = total_time / count if count else 0
    print(f"Average time: {avg_time:.5f} sec | Avg TFLOPS: {(2 * M * N * K) / avg_time / 1e12:.2f}")
