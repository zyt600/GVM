#include "memory_manager.h"
#include "../libgvmdrv/gvmdrv.h"
#include <exception>
#include <iostream>
#include <string>

MemoryManager &MemoryManager::getInstance() {
  static MemoryManager instance;
  instance.init();
  return instance;
}

bool MemoryManager::init() {
  std::cout << "[MemoryManager] Initializing memory manager" << std::endl;
  if (g_gvm_inited) {
    return true;
  }

  if (!initMemoryLimit()) {
    return false;
  }

  initUVMConnection();
  g_gvm_inited = true;
  return true;
}

bool MemoryManager::initMemoryLimit() {
  // Read memory limit from environment variable
  const char *env_limit = std::getenv(kMemoryLimitEnvVar);
  size_t memory_limit_gb = kDefaultMemoryLimitGB;

  if (env_limit != nullptr) {
    try {
      memory_limit_gb = std::stoull(env_limit);
      std::cout << "[GVM] Using memory limit from environment: "
                << memory_limit_gb << "GB" << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "[GVM] Invalid memory limit in environment variable, using "
                   "default: "
                << kDefaultMemoryLimitGB << "GB" << std::endl;
      memory_limit_gb = kDefaultMemoryLimitGB;
    }
  } else {
    std::cout << "[GVM] Using default memory limit: " << memory_limit_gb << "GB"
              << std::endl;
  }

  g_memory_limit =
      memory_limit_gb * 1024LL * 1024LL * 1024LL; // Convert GB to bytes
  return true;
}

bool MemoryManager::initUVMConnection() {
  // Initialize UVM connection
  g_uvm_fd = gvm_find_initialized_uvm();
  if (g_uvm_fd < 0) {
    std::cerr
        << "[GVM] Failed to find initd UVM, memory limiting may not work"
        << std::endl;
    return false;
  }

  // Set the memory limit via libgvmdrv
  gvm_set_gmemcg(g_uvm_fd, g_memory_limit);
  std::cout << "[GVM] Successfully set GPU memory limit to "
            << g_memory_limit / (1024LL * 1024LL * 1024LL) << "GB via libgvmdrv"
            << std::endl;

  return true;
}

bool MemoryManager::canAllocate(size_t size) {
  // Use GVM memory limit if available, otherwise fall back to GPU memory query
  int64_t effective_limit = g_memory_limit;
  if (effective_limit == 0) {
    // This should be set externally when we have GPU memory info
    effective_limit = g_cuda_mem_total;
  }

  if (g_cuda_mem_allocated + size > effective_limit) {
    if (g_allow_next_overcommit.load() &&
        g_failed_allocation_size.load() == size) {
      // This is a retry after garbage collection with the same size - allow
      // overcommit
      g_allow_next_overcommit = false;
      g_failed_allocation_size = 0;
      std::cout << "[MemoryManager] Allowing overcommit on retry. Requested: "
                << size / 1024 / 1024 << "MB, would exceed limit by: "
                << (g_cuda_mem_allocated + size - effective_limit) / 1024 / 1024
                << "MB" << std::endl;
      return true;
    } else {
      // First attempt or different size - return false to trigger PyTorch
      // garbage collection
      g_allow_next_overcommit = true;
      g_failed_allocation_size = size;
      std::cerr << "[MemoryManager] Out of memory (triggering GC). Requested: "
                << size / 1024 / 1024 << "MB, Available: "
                << (effective_limit - g_cuda_mem_allocated) / 1024 / 1024
                << "MB" << std::endl;
      return false;
    }
  }

  // Within limits - allow allocation
  return true;
}

void MemoryManager::recordAllocation(void *ptr, size_t size) {
  g_cuda_mem_map[ptr] = size;
  g_cuda_mem_allocated += size;

  int64_t effective_limit =
      g_memory_limit > 0 ? g_memory_limit : g_cuda_mem_total;
  std::cout << "[MemoryManager] Total CUDA memory allocated: "
            << g_cuda_mem_allocated / 1024 / 1024 << "MB / "
            << effective_limit / 1024 / 1024 << "MB" << std::endl;
}

size_t MemoryManager::recordDeallocation(void *ptr) {
  auto it = g_cuda_mem_map.find(ptr);
  if (it != g_cuda_mem_map.end()) {
    size_t size = it->second;
    g_cuda_mem_map.erase(it);
    g_cuda_mem_allocated -= size;
    std::cout << "[MemoryManager] Freed " << size / 1024 / 1024 << "MB, "
              << "remaining: " << g_cuda_mem_allocated / 1024 / 1024 << "MB"
              << std::endl;
    return size;
  }
  return 0;
}

void MemoryManager::getMemoryInfo(size_t *free, size_t *total,
                                  size_t actual_free, size_t actual_total) {
  // Store the actual GPU memory total for fallback
  if (g_cuda_mem_total == 0) {
    g_cuda_mem_total = actual_total;
  }

  // If we have a GVM memory limit set, report it as the total
  if (g_memory_limit > 0) {
    *total = g_memory_limit;
    int64_t free_mem = g_memory_limit - g_cuda_mem_allocated;
    *free = (free_mem < 0) ? 0 : static_cast<size_t>(free_mem);
  } else {
    // Use actual GPU memory info
    *total = actual_total;
    *free = actual_free;
  }
}