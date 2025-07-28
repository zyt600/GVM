
#ifndef SCHEDULER_LIBRARY_H
#define SCHEDULER_LIBRARY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <glib.h>
#include <stdatomic.h>
#include <stdbool.h> 

#include "../cuda-interception/nvml_subset.h"
#include "../cuda-interception/cuda_helper.h"

/**
 * Shared memory name for online kernel count.
 */
#define SHM_NAME "/online_kernel_flag"
#define SHM_SIZE sizeof(atomic_int)

/**
 * Proc file path for driver version
 */
#define DRIVER_VERSION_PROC_PATH "/proc/driver/nvidia/version"

/**
 * Driver regular expression pattern
 */
#define DRIVER_VERSION_MATCH_PATTERN "([0-9]+)(\\.[0-9]+)+"

/**
 * Max sample pid size
 */
#define MAX_PIDS (1024)

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define ROUND_UP(n, base) ((n) % (base) ? (n) + (base) - (n) % (base) : (n))

#define BUILD_BUG_ON(condition) ((void)sizeof(char[1 - 2 * !!(condition)]))

#define CAS(ptr, old, new) __sync_bool_compare_and_swap_8((ptr), (old), (new))
#define UNUSED __attribute__((unused))

#define MILLISEC (1000UL * 1000UL)

#define TIME_TICK (10)
#define FACTOR (32)
#define MAX_UTILIZATION (100)
#define CHANGE_LIMIT_INTERVAL (30)
#define USAGE_THRESHOLD (5)

#define GET_VALID_VALUE(x) (((x) >= 0 && (x) <= 100) ? (x) : 0)
#define CODEC_NORMALIZE(x) (x * 85 / 100)

#define DLSYM_HOOK_FUNC(f)                                       \
    if (0 == strcmp(symbol, #f)) {                               \
        return (void*) f; }                                      \

typedef struct {
  void *fn_ptr;
  char *name;
} entry_t;

typedef struct {
  int major;
  int minor;
} __attribute__((packed, aligned(8))) version_t;

typedef enum {
  INFO = 0,
  ERROR = 1,
  WARNING = 2,
  FATAL = 3,
  VERBOSE = 4,
} log_level_enum_t;

typedef struct {
  const char *func_name;      // base func name（like "cuGraphAddDependencies"）
  int min_ver;    // adjust to low version
  int max_ver;    // adjust to high version
  const char *real_name;      // the real name（ "cuGraphAddDependencies_v2"）
} CudaFuncMapEntry;

#define LOGGER(level, format, ...)                              \
  ({                                                            \
    char *_print_level_str = getenv("LOGGER_LEVEL");            \
    int _print_level = 3;                                       \
    if (_print_level_str) {                                     \
      _print_level = (int)strtoul(_print_level_str, NULL, 10);  \
      _print_level = _print_level < 0 ? 3 : _print_level;       \
    }                                                           \
    if (level <= _print_level) {                                \
      fprintf(stderr, "%s:%d " format "\n", __FILE__, __LINE__, \
              ##__VA_ARGS__);                                   \
    }                                                           \
    if (level == FATAL) {                                       \
      exit(-1);                                                 \
    }                                                           \
  })

/**
 * Load library and initialize some data
 */
void load_necessary_data();

extern entry_t cuda_library_entry[];
extern entry_t nvml_library_entry[];

// Global variables used in the scheduler.
extern GAsyncQueue *kernel_queue;
extern GAsyncQueue *events_queue;
extern pthread_once_t queues_init_once; 
extern pthread_once_t spawn_scheduler_thread_once;
extern pthread_once_t spawn_check_events_thread_once;
extern pthread_once_t init_flag_once;
extern atomic_int *online_kernel_count;
extern bool process_is_online;

// External declarations for functions defined in scheduler.c.
extern CUresult cuInit(unsigned int Flags);
extern CUresult cuDriverGetVersion(int *driverVersion);
extern CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);
extern CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);
extern CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                              unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
extern CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
extern CUresult cuLaunch(CUfunction f);
extern CUresult cuLaunchCooperativeKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                              unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                              unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                                              void **kernelParams);
extern CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                         unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                         unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                                         void **kernelParams);
extern CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
extern CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);
extern CUresult cuStreamSynchronize(CUstream hStream);
extern CUresult cuStreamSynchronize_ptsz(CUstream hStream);
extern CUresult cuCtxSynchronize(void);
extern CUresult cuEventSynchronize(CUevent hEvent);
extern CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
extern CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
extern CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);

#ifdef __cplusplus
}
#endif

#endif
