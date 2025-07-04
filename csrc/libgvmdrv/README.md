# libgvmdrv

A minimalistic interface for NVIDIA GPU Virtual Memory (UVM) driver with GVM extended functionality. This library provides C-compatible functions for interacting with NVIDIA GPU scheduling and memory management features.

## Features

- Find initialized UVM instances
- Control GPU timeslice allocation
- Manage GPU preemption
- Control channel scheduling
- Set interleave levels
- Memory management controls

## Requirements

- Linux system with NVIDIA drivers
- C++17 compatible compiler (GCC 7+ or Clang 5+)
- NVIDIA GPU with UVM support
- Root access for some operations

## Building

### Quick Build

```bash
make
```

This will build both shared and static libraries.

### Debug Build

```bash
make debug
```

### Build Options

- `make shared` - Build shared library only
- `make static` - Build static library only
- `make test` - Build and run test program
- `make clean` - Remove build artifacts

## Installation

### System-wide Installation

```bash
sudo make install
```

This installs the library to `/usr/local/lib` and headers to `/usr/local/include`.

### Custom Installation Path

```bash
make install PREFIX=/opt/libgvmdrv
```

## Usage

### C++ Example

```cpp
#include "libgvmdrv.h"
#include <iostream>

int main() {
    // Find initialized UVM
    int fd = find_initialized_uvm();
    if (fd < 0) {
        std::cerr << "Failed to find UVM" << std::endl;
        return 1;
    }

    // Get current timeslice
    long long unsigned timeslice = get_timeslice(fd);
    std::cout << "Current timeslice: " << timeslice << " us" << std::endl;

    // Set new timeslice
    set_timeslice(fd, timeslice * 2);

    return 0;
}
```

### C Example

```c
#include "libgvmdrv.h"
#include <stdio.h>

int main() {
    int fd = find_initialized_uvm();
    if (fd < 0) {
        fprintf(stderr, "Failed to find UVM\n");
        return 1;
    }

    // Set interleave level
    set_interleave(fd, 2);

    return 0;
}
```

### Compilation

```bash
# With shared library
g++ -o myapp myapp.cpp -lgvmdrv

# With static library
g++ -o myapp myapp.cpp /usr/local/lib/libgvmdrv.a

# Using pkg-config
g++ -o myapp myapp.cpp $(pkg-config --cflags --libs libgvmdrv)
```

## API Reference

### Core Functions

- `int find_initialized_uvm()` - Find and return file descriptor for initialized UVM
- `void set_timeslice(int fd, long long unsigned timesliceUs)` - Set GPU timeslice
- `long long unsigned get_timeslice(int fd)` - Get current GPU timeslice
- `void preempt(int fd)` - Preempt GPU execution
- `void restart(int fd)` - Restart GPU runlist
- `void schedule(int fd, bool enable)` - Enable/disable GPU scheduling
- `void stop(int fd)` - Stop GPU channel
- `void set_interleave(int fd, unsigned int interleave)` - Set interleave level
- `void bind(int fd)` - Bind GPU channel
- `void set_gmemcg(int fd, size_t size)` - Set GPU memory cgroup size

## Testing

Run the test program to verify functionality:

```bash
make test
./test
```

## Troubleshooting

### UVM Not Found

If `find_initialized_uvm()` returns -1:

1. Ensure NVIDIA drivers are loaded:
   ```bash
   sudo nvidia-modprobe -u -c=0
   ```

2. Check if UVM module is loaded:
   ```bash
   lsmod | grep nvidia_uvm
   ```

3. Verify device exists:
   ```bash
   ls -la /dev/nvidia-uvm
   ```

### Permission Denied

Some operations require root privileges. Run your application with sudo or ensure proper permissions.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]