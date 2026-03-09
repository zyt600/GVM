#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <signal.h>
#include <inttypes.h>

#include <gvm_notify.h>

#define ALLOC_SIZE (500ULL * 1024 * 1024)  // 500 MB
#define ALLOC_INTERVAL_US 500000           // 0.5 s = 500 ms
#define MAX_ALLOCS 4096
#define TOUCH_STRIDE (64 * 1024)           // touch one byte per 64 KB page

static volatile int g_stop = 0;
static int g_ignore_notice = 0;
static int g_no_touch = 0;
static pthread_mutex_t g_alloc_lock = PTHREAD_MUTEX_INITIALIZER;
static void *g_ptrs[MAX_ALLOCS];
static int g_count = 0;

static uint64_t read_uvm_file(const char *filename) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/kernel/debug/nvidia-uvm/processes/%d/0/%s",
             (int)getpid(), filename);
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    uint64_t val = 0;
    fscanf(f, "%" SCNu64, &val);
    fclose(f);
    return val;
}

static uint64_t read_memory_current(void) {
    return read_uvm_file("memory.current");
}

static uint64_t read_swap_current(void) {
    return read_uvm_file("memory.swap.current");
}

#define GB(x) ((double)(x) / (1024.0 * 1024.0 * 1024.0))

static void sigint_handler(int sig) {
    (void)sig;
    g_stop = 1;
}

static void shrink_to_target(uint64_t target, uint64_t current) {
    int freed = 0;
    while (g_count > 0 && current > target) {
        --g_count;
        cudaFree(g_ptrs[g_count]);
        g_ptrs[g_count] = NULL;
        current = (current > ALLOC_SIZE) ? current - ALLOC_SIZE : 0;
        freed++;
    }
    printf("[listener] Freed %d blocks, total alloc now=%.2f GB, phys now=%.2f GB, swap now=%.2f GB (target=%.2f GB)\n",
           freed, GB(current), GB(read_memory_current()), GB(read_swap_current()), GB(target));
    fflush(stdout);
}

__global__ void touch_kernel(volatile char *ptr, size_t len, size_t stride) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t pos = idx * stride;
    if (pos < len)
        ptr[pos] += 1;
}

static void *memory_toucher(void *arg) {
    (void)arg;
    while (!g_stop) {
        pthread_mutex_lock(&g_alloc_lock);
        int n = g_count;
        for (int i = 0; i < n && !g_stop; i++) {
            if (!g_ptrs[i])
                continue;
            size_t num_touches = ALLOC_SIZE / TOUCH_STRIDE;
            int threads = 256;
            int blocks = (num_touches + threads - 1) / threads;
            touch_kernel<<<blocks, threads>>>((volatile char *)g_ptrs[i], ALLOC_SIZE, TOUCH_STRIDE);
        }
        cudaDeviceSynchronize();
        pthread_mutex_unlock(&g_alloc_lock);

        uint64_t swp = read_swap_current();
        if (swp > 0) {
            printf("[toucher] Touched %d blocks, phys=%.2f GB, swap=%.2f GB\n",
                   n, GB(read_memory_current()), GB(swp));
            fflush(stdout);
        }
        sleep(3);
    }
    return NULL;
}

static void notice_handler(const UVM_WAIT_NOTICE_PARAMS *params) {
    switch (params->type) {
    case GVM_NOTICE_EVICTION:
        printf("[eviction] *** SHRINK NOTICE: target=%.2f GB, current=%.2f GB, phys=%.2f GB, swap=%.2f GB ***\n",
               GB(params->eviction.target_memory), GB(params->eviction.current_memory),
               GB(read_memory_current()), GB(read_swap_current()));
        fflush(stdout);

        if (g_ignore_notice) {
            printf("[eviction] --ignore-notice set, NOT shrinking (waiting for kernel force eviction)\n");
            fflush(stdout);
        } else {
            pthread_mutex_lock(&g_alloc_lock);
            shrink_to_target(params->eviction.target_memory, params->eviction.current_memory);
            pthread_mutex_unlock(&g_alloc_lock);
        }
        break;
    case GVM_NOTICE_AVAILABILITY: {
        uint64_t current = read_memory_current();
        printf("[available] *** MEMORY AVAILABLE: available=%.2f GB, current=%.2f GB, swap=%.2f GB ***\n",
               GB(params->availability.available_memory), GB(current), GB(read_swap_current()));
        fflush(stdout);
        break;
    }
    }
}

static void *memory_monitor(void *arg) {
    (void)arg;
    while (!g_stop) {
        uint64_t cur = read_memory_current();
        uint64_t swp = read_swap_current();
        printf("[monitor] memory.current=%.2f GB, swap=%.2f GB, allocs=%d\n",
               GB(cur), GB(swp), g_count);
        fflush(stdout);
        sleep(2);
    }
    return NULL;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ignore-notice") == 0)
            g_ignore_notice = 1;
        else if (strcmp(argv[i], "--no-touch") == 0)
            g_no_touch = 1;
        else{
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    if (g_ignore_notice)
        printf("[main] --ignore-notice mode: will NOT cooperate with shrink notices\n");
    if (g_no_touch)
        printf("[main] --no-touch mode: memory_toucher thread disabled\n");

    signal(SIGINT, sigint_handler);

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    void *dummy;
    err = cudaMalloc(&dummy, 1024);
    if (err != cudaSuccess) {
        fprintf(stderr, "Initial cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaFree(dummy);
    printf("[main] CUDA context initialized on GPU 0 (pid=%d)\n", (int)getpid());

    if (gvm_register_notify(notice_handler) != 0) {
        fprintf(stderr, "gvm_register_notify failed\n");
        return 1;
    }

    pthread_t mon_tid;
    if (pthread_create(&mon_tid, NULL, memory_monitor, NULL) != 0) {
        perror("pthread_create monitor");
        return 1;
    }

    pthread_t touch_tid;
    if (!g_no_touch) {
        if (pthread_create(&touch_tid, NULL, memory_toucher, NULL) != 0) {
            perror("pthread_create memory_toucher");
            return 1;
        }
    }

    size_t gpu_free = 0, gpu_total = 0;
    err = cudaMemGetInfo(&gpu_free, &gpu_total);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("[main] GPU total memory: %.2f GB\n",
           (double)gpu_total / (1024.0 * 1024.0 * 1024.0));

    usleep(500000);

    printf("[main] Starting allocation loop: %llu MB every %.0f ms (Ctrl+C to stop)\n",
           (unsigned long long)(ALLOC_SIZE / (1024ULL * 1024)),
           (double)ALLOC_INTERVAL_US / 1000.0);
    fflush(stdout);

    struct timespec ts = { .tv_sec = 0, .tv_nsec = ALLOC_INTERVAL_US * 1000 };

    while (!g_stop) {
        pthread_mutex_lock(&g_alloc_lock);

        if (g_count >= MAX_ALLOCS) {
            pthread_mutex_unlock(&g_alloc_lock);
            nanosleep(&ts, NULL);
            continue;
        }

        if (read_memory_current() + ALLOC_SIZE > gpu_total) {
            pthread_mutex_unlock(&g_alloc_lock);
            printf("[main] Stop allocating: phys %.2f GB + next %.2f GB > GPU total %.2f GB\n",
                   GB(read_memory_current()), GB((uint64_t)ALLOC_SIZE), GB(gpu_total));
            fflush(stdout);
            sleep(1);
            continue;
        }

        err = cudaMalloc(&g_ptrs[g_count], ALLOC_SIZE);
        if (err != cudaSuccess) {
            pthread_mutex_unlock(&g_alloc_lock);
            printf("[main] cudaMalloc failed (memory.current=%.2f GB, swap=%.2f GB): %s, waiting...\n",
                   GB(read_memory_current()), GB(read_swap_current()),
                   cudaGetErrorString(err));
            nanosleep(&ts, NULL);
            continue;
        }

        cudaMemset(g_ptrs[g_count], 0xAB, ALLOC_SIZE);
        g_count++;

        if (g_count % 10 == 0) {
            printf("[main] Allocated %d x 500MB, memory.current=%.2f GB, swap=%.2f GB\n",
                   g_count, GB(read_memory_current()), GB(read_swap_current()));
            fflush(stdout);
        }

        pthread_mutex_unlock(&g_alloc_lock);
        nanosleep(&ts, NULL);
    }

    printf("[main] Stopping. Exiting immediately.\n");
    fflush(stdout);
    _exit(0);
}
