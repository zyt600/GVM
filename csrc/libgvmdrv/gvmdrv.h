/**
 * @file libgvmdrv.h
 * @brief A minimalistic interface set for interacting with the NVIDIA GPU
 * Virtual Memory (UVM) driver with GVM extended functionality. Working with
 * both C and C++ code.
 */

#ifndef __LIBGVMDRV_H__
#define __LIBGVMDRV_H__

#ifdef __cplusplus
extern "C" {
#endif

int find_initialized_uvm();
void set_timeslice(int fd, long long unsigned timesliceUs);
long long unsigned get_timeslice(int fd);
void preempt(int fd);
void restart(int fd);
void schedule(int fd, bool enable);
void stop(int fd);
void set_interleave(int fd, unsigned int interleave);
void bind(int fd);
void set_gmemcg(int fd, unsigned long long size);

#ifdef __cplusplus
}
#endif

#endif // __LIBGVMDRV_H__
