#!/usr/bin/env bash

# Generate entries for loader.c.

# Set path to your entry file, e.g., entry_file.h
ENTRY_FILE="entry_file.h"

grep -oP 'CUDA_ENTRY_ENUM\(\K[^\)]+' "$ENTRY_FILE" | while read -r symbol; do
    echo "    {.name = \"$symbol\"},"
done