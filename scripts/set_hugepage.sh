#!/bin/bash

op=${1:-enable}

if [ "$op" == "enable" ]; then
    echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
    echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
elif [ "$op" == "disable" ]; then
    echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
    echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
else
    echo "Usage: $0 [enable|disable]"
    exit 1
fi
