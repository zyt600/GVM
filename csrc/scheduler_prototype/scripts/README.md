The scripts in this folder can be utilized to generate code and identify new cuda driver APIs in future versions.

# find_new_lib.sh

Use this script to identify any missing cuda or nvml declarations like so:

```
./find_new_lib.sh /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libnvidia-ml.so
```

# generate_cuda_name_list.sh

Extracts CUDA entry symbols from a header file and prints them in a format suitable for loader.c.
Edit the `ENTRY_FILE` variable to point to your entry file (e.g., `entry_file.h`). Each entry in the entry file
should look something like: `CUDA_ENTRY_ENUM(cuCtxGetSharedMemConfig),`

```
./generate_cuda_name_list.sh
```

# generate_nvml_name_list.sh

Extracts NVML entry symbols from a header file and prints them in a format suitable for loader.c.
Edit the `ENTRY_FILE` variable to point to your NVML entry file (e.g., `nvml_entry_file.h`). Each entry in the entry file
should look something like: `NVML_ENTRY_ENUM(nvmlInit),`

```
./generate_nvml_name_list.sh
```

# generate_cuda_wrappers.py

Generates wrapper functions for CUDA APIs by parsing entries and function signatures from header files.
Edit `ENTRY_FILE` and `HEADER_FILE` variables as needed. The entry file contains entries in the form of `CUDA_ENTRY_ENUM(...)`
while the header file contains `Uresult CUDAAPI ...` declarations.

```
python3 generate_cuda_wrappers.py
```

# generate_header_file.py

Extracts function declarations from a C source file and writes them to a header file.

```
python3 generate_header_file.py <input_file.c> <output_header_file.h>
```

# generate_dlsym_hooks.py

Generates DLSYM_HOOK_FUNC macros for each CUDA entry symbol found in the input file.

```
python3 generate_dlsym_hooks.py <input_file.txt>
```