#!/usr/bin/env python3

import sys
import re

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_dlsym_hooks.py <input_file.txt>")
        sys.exit(1)

    input_file = sys.argv[1]

    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Regex to extract the symbol inside CUDA_ENTRY_ENUM(...)
    pattern = re.compile(r'CUDA_ENTRY_ENUM\(\s*(\w+)\s*\)')

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        match = pattern.search(line)
        if match:
            symbol = match.group(1)
            print(f"DLSYM_HOOK_FUNC({symbol});")
        else:
            pass
            # Optionally warn if no match
            # print(f"// Warning: No match in line: {line}")

if __name__ == "__main__":
    main()