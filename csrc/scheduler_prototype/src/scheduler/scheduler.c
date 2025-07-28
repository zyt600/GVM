#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/un.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <string.h>
#include <fcntl.h> 
#include <sys/mman.h>

#include "../../include/cuda-interception/cuda_helper.h"
#include "../../include/scheduler/scheduler.h"
#include "../../include/cuda-interception/nvml_helper.h"
#include "../../include/cuda-interception/dlsym_hook.h"

struct kernel {
  CUfunction f;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  void **kernelParams;
  void **extra;
  CUcontext context;
  GAsyncQueue *completion_queue;
};

GAsyncQueue *kernel_queue = NULL;
GAsyncQueue *events_queue = NULL;
pthread_once_t queues_init_once = PTHREAD_ONCE_INIT; 
pthread_once_t spawn_scheduler_thread_once = PTHREAD_ONCE_INIT;
pthread_once_t spawn_check_events_thread_once = PTHREAD_ONCE_INIT;
pthread_once_t init_flag_once = PTHREAD_ONCE_INIT;
atomic_int *online_kernel_count = NULL;
bool process_is_online = NULL;
// TODO (Tony): Code for asynchronous version.
// static atomic_int pending_kernels = 0;

void init_shared_flag() {
    fprintf(stderr, "[Scheduler] Initializing shared atomic_bool for online running flag.\n");
    
    int shm_fd;
    int created = 0;

    // Try to create the shared memory object exclusively.
    shm_fd = shm_open(SHM_NAME, O_CREAT | O_EXCL | O_RDWR, 0666);
    if (shm_fd == -1) {
        if (errno == EEXIST) {
            // Already exists — open without O_EXCL.
            shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
            if (shm_fd == -1) {
                perror("shm_open (existing)");
                exit(EXIT_FAILURE);
            }
        } else {
            perror("shm_open (create)");
            exit(EXIT_FAILURE);
        }
    } else {
        // We created it, need to initialize it.
        created = 1;
    }

    // Only truncate if we're the creator.
    if (created) {
        if (ftruncate(shm_fd, SHM_SIZE) == -1) {
            perror("ftruncate");
            exit(EXIT_FAILURE);
        }
    }

    // Map the shared memory into this process’s address space
    void *ptr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    // Cast the pointer to our shared atomic_int
    online_kernel_count = (atomic_int *)ptr;

    // Only initialize the flag if we created it.
    if (created) {
      atomic_store(online_kernel_count, 0);
    }

    fprintf(stderr, "[Scheduler] Shared atomic_int initialized at %p with initial value %d.\n",
        online_kernel_count, atomic_load(online_kernel_count));
}

static void init_queues() {
  fprintf(stderr, "[Scheduler] Initializing kernel queue (pid %d thread %lu)\n",
          getpid(),
          (unsigned long)syscall(SYS_gettid));

  if (kernel_queue == NULL) {
    kernel_queue = g_async_queue_new();
    if (!kernel_queue) {
      fprintf(stderr, "Failed to create kernel queue\n");
      exit(EXIT_FAILURE);
    }
  }

  if (events_queue == NULL) {
    events_queue = g_async_queue_new();
    if (!events_queue) {
      fprintf(stderr, "Failed to create events queue\n");
      exit(EXIT_FAILURE);
    }
  }
}

static void* run_scheduler(void *arg) {
  fprintf(stderr, "[Scheduler Thread] pid %d thread %lu started\n",
          getpid(),
          (unsigned long)syscall(SYS_gettid));

  while (1) {
    // Block until a kernel is available in the queue.
    struct kernel *k = g_async_queue_pop(kernel_queue);
    if (!k) {
        fprintf(stderr, "[Scheduler] No kernel to process, exiting thread.\n");
        exit(EXIT_SUCCESS);
    }

    // Check shared variable to see if we can launch this kernel.
    if (process_is_online) {
      atomic_fetch_add(online_kernel_count, 1);
    } else {
      // If the kernel is offline and the online kernel count is greater than 0, we need to wait
      // until all online kernels have completed.
      while (atomic_load(online_kernel_count) > 0) {
        fprintf(stderr, "[Scheduler] Kernel is offline, waiting for online running flag to be cleared.\n");
        sched_yield();
      }
    }

    // Set the current context to the one stored in the kernel structure.
    CUresult set_ctx_result = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxSetCurrent, k->context);
    if (set_ctx_result != CUDA_SUCCESS) {
        fprintf(stderr, "[Scheduler] Failed to set current context: %d\n", set_ctx_result);
        exit(EXIT_FAILURE);
    }

    CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel,
                                    k->f,
                                    k->gridDimX, k->gridDimY, k->gridDimZ,
                                    k->blockDimX, k->blockDimY, k->blockDimZ,
                                    k->sharedMemBytes,
                                    k->hStream,
                                    k->kernelParams,
                                    k->extra);

    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "[Scheduler] Kernel launch failed with error: %d\n", result);
        exit(EXIT_FAILURE);
    }

    // We must also create an event for this kernel and add it to the events queue.
    if (process_is_online) {
        CUevent event;
        CUresult evt_create_res = CUDA_ENTRY_CALL(cuda_library_entry, cuEventCreate, &event, CU_EVENT_DEFAULT);
        if (evt_create_res != CUDA_SUCCESS) {
            fprintf(stderr, "[Scheduler] Failed to create event: %d\n", evt_create_res);
            exit(EXIT_FAILURE);
        }

        CUresult evt_record_res = CUDA_ENTRY_CALL(cuda_library_entry, cuEventRecord, event, k->hStream);
        if (evt_record_res != CUDA_SUCCESS) {
            fprintf(stderr, "[Scheduler] Failed to record event: %d\n", evt_record_res);
            exit(EXIT_FAILURE);
        }

        // Push the event into the event queue for monitoring
        g_async_queue_push(events_queue, event);
    }

    // TODO (Tony): Code for asynchronous version.
    // atomic_fetch_sub(&pending_kernels, 1);

    // Signal successful launch by pushing a message to the completion queue.
    g_async_queue_push(k->completion_queue, (void*)1);
    free(k);
  }
  
  return NULL;
}

static void spawn_scheduler_thread() {
  pthread_t scheduler_thread;
  if (pthread_create(&scheduler_thread, NULL, run_scheduler, NULL) != 0) {
      fprintf(stderr, "Failed to create scheduler thread\n");
  } else {
      pthread_detach(scheduler_thread);
  }
}

static void* check_events(void *arg) {
  fprintf(stderr, "[Check Events Thread] pid %d thread %lu started\n",
          getpid(),
          (unsigned long)syscall(SYS_gettid));

  while (1) {
    CUevent event = (CUevent) g_async_queue_pop(events_queue); // Blocks until an event is available.

    // Poll the event until it completes.
    while (1) {
      CUresult status = CUDA_ENTRY_CALL(cuda_library_entry, cuEventQuery, event);
      if (status == CUDA_SUCCESS) {
        break; // Event has completed.
      } else if (status == CUDA_ERROR_NOT_READY) {
        sched_yield(); // Event is not ready, yield and check again.
      } else {
        fprintf(stderr, "[Check Events] cuEventQuery failed: %d\n", status);
        break;
      }
    }

    // Destroy event to free GPU-side resources.
    CUresult destroy_status  = CUDA_ENTRY_CALL(cuda_library_entry, cuEventDestroy, event);
    if (destroy_status != CUDA_SUCCESS) {
      fprintf(stderr, "[Check Events] cuEventDestroy failed: %d\n", destroy_status);
    }

    // Decrement the online kernel count.
    int prev = atomic_fetch_sub(online_kernel_count, 1);
    if (prev <= 0) {
      fprintf(stderr, "[Check Events] Warning: online_running_count dropped below zero!\n");
      exit(EXIT_FAILURE);
    }
  }
}

static void spawn_check_events_thread() {
  pthread_t check_events_thread;
  if (pthread_create(&check_events_thread, NULL, check_events, NULL) != 0) {
      fprintf(stderr, "Failed to create check events thread\n");
  } else {
      pthread_detach(check_events_thread);
  }
}

// TODO (Tony): Code for asynchronous version.
// static void wait_for_all_kernels() {
//     while (atomic_load(&pending_kernels) > 0) {
//         sched_yield();
//     }

//     fprintf(stderr, "[Scheduler] All kernels have completed.\n");
// }

/** hijack entrypoint */
CUresult cuDriverGetVersion(int *driverVersion) {
  CUresult ret;

  load_necessary_data();

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuDriverGetVersion, driverVersion);
  return ret;
}

CUresult cuInit(unsigned int Flags) {
  CUresult ret;

  fprintf(stderr, "[cuInit] pid %d thread %lu\n",
          getpid(),
          (unsigned long)syscall(SYS_gettid));

  load_necessary_data();

  // Initialize kernel queue and event queue before creating thread.
  pthread_once(&queues_init_once, init_queues);

  // Initialize the shared flag variable if needed.
  pthread_once(&init_flag_once, init_shared_flag);

  fprintf(stderr, "[cuInit] Initializing CUDA with flags: %u\n", Flags);

  ret = CUDA_ENTRY_CALL(cuda_library_entry, cuInit, Flags);
  if (ret != CUDA_SUCCESS) {
    return ret;
  }

  fprintf(stderr, "[cuInit] cuInit completed successfully.\n");

  // Read environment variable to determine if the kernel is online or offline.
  const char *online_env = getenv("ONLINE_KERNEL");
  if (online_env && strcmp(online_env, "1") == 0) {
      fprintf(stderr, "[cuInit] Kernel is set to online mode.\n");
      process_is_online = true;
  } else {
      fprintf(stderr, "[cuInit] Kernel is set to offline mode.\n");
      process_is_online = false;
  }

  // Ensure scheduler thread is created only once.
  pthread_once(&spawn_scheduler_thread_once, spawn_scheduler_thread);

  // Ensure events thread is created only once.
  pthread_once(&spawn_check_events_thread_once, spawn_check_events_thread);

  return ret;
}

extern nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
  load_necessary_data();
  fprintf(stderr, "[nvmlInitWithFlags] flags: %u\n", flags);

  return NVML_ENTRY_CALL(nvml_library_entry, nvmlInitWithFlags, flags);
}

nvmlReturn_t nvmlInit_v2(void) {
  load_necessary_data();
  fprintf(stderr, "[nvmlInit_v2] Initializing NVML v2\n");

  return NVML_ENTRY_CALL(nvml_library_entry, nvmlInit_v2);
}

// if multi func, we can add here
// like 12030，means cuda 12.3 ，cuda.h header may give start at version
// all new add func put here
static CudaFuncMapEntry g_func_map[] = {
    {"cuGraphAddKernelNode", 10000, 11999, "cuGraphAddKernelNode"},
    {"cuGraphAddKernelNode", 12000, 99999, "cuGraphAddKernelNode_v2"},

    {"cuGraphKernelNodeGetParams", 10000, 11999, "cuGraphKernelNodeGetParams"},
    {"cuGraphKernelNodeGetParams", 12000, 99999, "cuGraphKernelNodeGetParams_v2"},

    {"cuGraphKernelNodeSetParams", 10000, 11999, "cuGraphKernelNodeSetParams"},
    {"cuGraphKernelNodeSetParams", 12000, 99999, "cuGraphKernelNodeSetParams_v2"}
};

// find func by cuda version
const char* get_real_func_name(const char* base_name,int cuda_version) {
  int i = 0;
  for (i = 0; i < sizeof(g_func_map)/sizeof(g_func_map[0]); ++i) {
    CudaFuncMapEntry *entry = &g_func_map[i];
    // check fun name
    if (strcmp(entry->func_name, base_name) != 0) continue;
    // check cuda version
    if (cuda_version >= entry->min_ver && cuda_version <= entry->max_ver) {
      return entry->real_name;
    }
  }
  return NULL; // if not found
}

void* find_real_symbols_in_table(const char *symbol) {
  void *pfn;
  // this symbol always has suffix like _v2,_v3
  pfn = __dlsym_hook_section(NULL,symbol);
  if (pfn!=NULL) {
    return pfn;
  }
  return NULL;
}

void *find_symbols_in_table(const char *symbol) {
    char symbol_v[500];
    void *pfn;
    strcpy(symbol_v,symbol);
    strcat(symbol_v,"_v3");
    pfn = __dlsym_hook_section(NULL,symbol_v);
    if (pfn!=NULL) {
        return pfn;
    }
    symbol_v[strlen(symbol_v)-1]='2';
    pfn = __dlsym_hook_section(NULL,symbol_v);
    if (pfn!=NULL) {
        return pfn;
    }
    pfn = __dlsym_hook_section(NULL,symbol);
    if (pfn!=NULL) {
        return pfn;
    }
    return NULL;
}

void *find_symbols_in_table_by_cudaversion(const char *symbol,int  cudaVersion) {
  void *pfn;
  const char *real_symbol;
  real_symbol = get_real_func_name(symbol, cudaVersion);
  if (real_symbol == NULL) {
    // if not find in mulit func version def, use origin logic
    pfn = find_symbols_in_table(symbol);
  } else {
    pfn = find_real_symbols_in_table(real_symbol);
  }
  return pfn;
}

CUresult (*cuGetProcAddress_real) (const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags); 

CUresult _cuGetProcAddress(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags) {
    fprintf(stderr, "[_cuGetProcAddress] symbol: %s, cudaVersion: %d, flags: %llu\n", 
           symbol, cudaVersion, (unsigned long long)flags);

    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (*pfn==NULL){
        CUresult res = CUDA_ENTRY_CALL(cuda_library_entry,cuGetProcAddress,symbol,pfn,cudaVersion,flags);
        return res;
    }else{
        return CUDA_SUCCESS;
    }
}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags) {
    fprintf(stderr, "[cuGetProcAddress] symbol: %s, cudaVersion: %d, flags: %llu\n", 
           symbol, cudaVersion, (unsigned long long)flags);

    // We need to call this because cuGetProcAddress can be the first function that's called where the symbol is "Init".
    load_necessary_data();  
    
    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (strcmp(symbol,"cuGetProcAddress")==0) {
        CUresult res = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress, symbol, pfn, cudaVersion, flags); 
        if (res==CUDA_SUCCESS) {
            cuGetProcAddress_real=*pfn;
            *pfn=_cuGetProcAddress;
        }
        return res;
    }
    if (*pfn==NULL){
        CUresult res = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress, symbol, pfn, cudaVersion, flags);
        return res;
    }else{
        return CUDA_SUCCESS;
    }
}

CUresult _cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus){
    fprintf(stderr, "[_cuGetProcAddress_v2] symbol: %s, cudaVersion: %d, flags: %llu\n", 
           symbol, cudaVersion, (unsigned long long)flags);
  
    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (*pfn==NULL){
        CUresult res = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, pfn, cudaVersion, flags, symbolStatus);
        return res;
    }else{
        return CUDA_SUCCESS;
    } 
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    // fprintf(stderr, "[cuGetProcAddress_v2] symbol: %s, cudaVersion: %d, flags: %llu\n", 
    //        symbol, cudaVersion, (unsigned long long)flags);

    // We need to call this because cuGetProcAddress_v2 can be the first function that's called where the symbol is "Init".
    load_necessary_data();  

    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (strcmp(symbol,"cuGetProcAddress_v2")==0) {
        CUresult res = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, pfn, cudaVersion, flags, symbolStatus); 
        if (res==CUDA_SUCCESS) {
            cuGetProcAddress_real=*pfn;
            *pfn=_cuGetProcAddress_v2;
        }
        return res;
    }
    if (*pfn==NULL){
        CUresult res = CUDA_ENTRY_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, pfn, cudaVersion, flags, symbolStatus);
        return res;
    }else{
        return CUDA_SUCCESS;
    }
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra) {
  struct kernel *k = malloc(sizeof(struct kernel));
  if (!k) {
      fprintf(stderr, "Failed to allocate memory for kernel structure\n");
      return CUDA_ERROR_OUT_OF_MEMORY; 
  }
  memset(k, 0, sizeof(struct kernel));

  k->f = f;
  k->gridDimX = gridDimX;
  k->gridDimY = gridDimY;
  k->gridDimZ = gridDimZ;
  k->blockDimX = blockDimX;
  k->blockDimY = blockDimY;
  k->blockDimZ = blockDimZ;
  k->sharedMemBytes = sharedMemBytes;
  k->hStream = hStream;
  // TODO: We should be able to parse this and make the code asynchronous.
  k->kernelParams = kernelParams;
  k->extra = extra;

  // TODO (Tony): Code for asynchronous version.
//   // Copy kernel parameters if they exist
//   if (kernelParams) {
//       // Count the number of parameters (assuming NULL-terminated)
//       size_t param_count = 0;
//       while (kernelParams[param_count] != NULL) {
//           param_count++;
//       }
      
//       // Allocate space for params array + terminating NULL
//       k->kernelParams = g_malloc0((param_count + 1) * sizeof(void*));
      
//       // Copy each parameter pointer
//       for (size_t i = 0; i < param_count; i++) {
//           k->kernelParams[i] = kernelParams[i];
//       }
//       k->kernelParams[param_count] = NULL; // Ensure NULL termination
//   }

//   // Copy extra parameters if they exist
//   if (extra) {
//       // Count the number of extra parameters (assuming NULL-terminated)
//       size_t extra_count = 0;
//       while (extra[extra_count] != NULL) {
//           extra_count++;
//       }
      
//       // Allocate space for extras array + terminating NULL
//       k->extra = g_malloc0((extra_count + 1) * sizeof(void*));
      
//       // Copy each extra pointer
//       for (size_t i = 0; i < extra_count; i++) {
//           k->extra[i] = extra[i];
//       }
//       k->extra[extra_count] = NULL; // Ensure NULL termination
//   }

  // Get the current context.
  CUcontext current_ctx;
  CUresult result = CUDA_ENTRY_CALL(cuda_library_entry, cuCtxGetCurrent, &current_ctx);
  if (result != CUDA_SUCCESS) {
      fprintf(stderr, "Failed to get current context: %d\n", result);
      exit(EXIT_FAILURE);
  }
  // Store the context in the kernel structure.
  k->context = current_ctx;

  // Create completion queue.
  GAsyncQueue *completion_queue = g_async_queue_new();
  if (!completion_queue) {
      fprintf(stderr, "Failed to create completion queue\n");
      free(k);
      return CUDA_ERROR_OUT_OF_MEMORY; // Return an error if queue creation fails.
  }
  k->completion_queue = completion_queue;

  // TODO (Tony): Code for asynchronous version.
  // atomic_fetch_add(&pending_kernels, 1);

  // Push the kernel onto the queue.
  g_async_queue_push(kernel_queue, k);

  // Wait for completion signal.
  g_async_queue_pop(completion_queue);

  g_async_queue_unref(completion_queue);

  return CUDA_SUCCESS;
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX,
                             unsigned int gridDimY, unsigned int gridDimZ,
                             unsigned int blockDimX, unsigned int blockDimY,
                             unsigned int blockDimZ,
                             unsigned int sharedMemBytes, CUstream hStream,
                             void **kernelParams, void **extra) {
  printf("[cuLaunchKernel_ptsz] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchKernel_ptsz, f, gridDimX,
                         gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunch(CUfunction f) {
  printf("[cuLaunch] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunch, f);
}

CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  printf("[cuLaunchCooperativeKernel_ptsz] thread %lu\n",
         (unsigned long)syscall(SYS_gettid));
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchCooperativeKernel_ptsz, f,
                         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                         blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                   unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream, void **kernelParams) {
  printf("[cuLaunchCooperativeKernel] thread %lu\n",
         (unsigned long)syscall(SYS_gettid));
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchCooperativeKernel, f,
                         gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                         blockDimZ, sharedMemBytes, hStream, kernelParams);
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  printf("[cuLaunchGrid] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGrid, f, grid_width,
                         grid_height);
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream) {
  printf("[cuLaunchGridAsync] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  return CUDA_ENTRY_CALL(cuda_library_entry, cuLaunchGridAsync, f, grid_width,
                         grid_height, hStream);
}

CUresult cuStreamSynchronize(CUstream hStream) {
  // printf("[cuStreamSynchronize] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  // TODO (Tony): Code for asynchronous version.
  // wait_for_all_kernels();
  return CUDA_ENTRY_CALL(cuda_library_entry, cuStreamSynchronize, hStream);
}

CUresult cuStreamSynchronize_ptsz(CUstream hStream) {
  // printf("[cuStreamSynchronize_ptsz] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  // TODO (Tony): Code for asynchronous version.
  // wait_for_all_kernels();
  return CUDA_ENTRY_CALL(cuda_library_entry, cuStreamSynchronize_ptsz, hStream);
}

CUresult cuCtxSynchronize(void) {
  // printf("[cuCtxSynchronize] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  // TODO (Tony): Code for asynchronous version.
  // wait_for_all_kernels();
  return CUDA_ENTRY_CALL(cuda_library_entry, cuCtxSynchronize);
}

CUresult cuEventSynchronize(CUevent hEvent) {
  // printf("[cuEventSynchronize] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  // TODO (Tony): Code for asynchronous version.
  // wait_for_all_kernels();
  return CUDA_ENTRY_CALL(cuda_library_entry, cuEventSynchronize, hEvent);
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  // printf("[cuMemcpy] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  // TODO (Tony): Code for asynchronous version.
  // wait_for_all_kernels();
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpy, dst, src, ByteCount);
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                         size_t ByteCount) {
  // printf("[cuMemcpyDtoH_v2] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  // TODO (Tony): Code for asynchronous version.
  // wait_for_all_kernels();
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoH_v2, dstHost,
                         srcDevice, ByteCount);
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  // printf("[cuMemcpyDtoH] thread %lu\n", (unsigned long)syscall(SYS_gettid));
  // TODO (Tony): Code for asynchronous version.
  // wait_for_all_kernels();
  return CUDA_ENTRY_CALL(cuda_library_entry, cuMemcpyDtoH, dstHost, srcDevice,
                         ByteCount);
}