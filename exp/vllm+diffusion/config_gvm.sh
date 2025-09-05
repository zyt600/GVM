#!/bin/bash

# Script to find VLLM process using GPU and set compute priority
# Usage: ./config_gvm.sh <VLLM_PRIORITY> <DIFFUSION_PRIORITY>

set -e

VLLM_PRIORITY=${1:-0}
DIFFUSION_PRIORITY=${2:-8}


set_vllm_priority() {
echo "Looking for VLLM processes using GPU..."

# Find VLLM processes that are using GPU
# We'll look for processes with "vllm" in the name that are using GPU memory
vllm_pids=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits | grep -i vllm | cut -d',' -f1 | tr -d ' ')

if [ -z "$vllm_pids" ]; then
    echo "No VLLM processes found using GPU"
else
    echo "Found VLLM processes with PIDs: $vllm_pids"

    # Process each PID
    for pid in $vllm_pids; do
    echo "Processing PID: $pid"

    # Check if the nvidia-uvm process directory exists
    uvm_path="/sys/kernel/debug/nvidia-uvm/processes/$pid/0/compute.priority"

    if [ ! -f "$uvm_path" ]; then
        echo "Warning: UVM path not found for PID $pid: $uvm_path"
        continue
    fi

    # Set the compute priority to $VLLM_PRIORITY for VLLM PID $pid
    echo "Setting compute priority to $VLLM_PRIORITY for VLLM PID $pid"
    echo 0 | sudo tee "$uvm_path" > /dev/null

    if [ $? -eq 0 ]; then
        echo "Successfully set compute priority to $VLLM_PRIORITY for VLLM PID $pid"
    else
        echo "Failed to set compute priority to $VLLM_PRIORITY for VLLM PID $pid"
    fi
    done

    echo "Done processing all VLLM processes"
fi
}

set_diffusion_priority() {
# Look for diffusion processes (python diffusion.py)
echo "Looking for Diffusion processes using GPU..."

# Find processes that contain "diffusion.py" in their command line
diffusion_pids=""
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')

for gpu_pid in $gpu_pids; do
    # Check if this PID is running python diffusion.py
    if ps -p "$gpu_pid" -o cmd --no-headers 2>/dev/null | grep -q "python.*diffusion\.py"; then
        diffusion_pids="$diffusion_pids $gpu_pid"
    fi
done

# Remove leading/trailing spaces
diffusion_pids=$(echo $diffusion_pids | xargs)

if [ -z "$diffusion_pids" ]; then
    echo "No Diffusion processes (python diffusion.py) found using GPU"
else
    echo "Found Diffusion processes with PIDs: $diffusion_pids"

    # Process each diffusion PID
    for pid in $diffusion_pids; do
        echo "Processing Diffusion PID: $pid"

        # Check if the nvidia-uvm process directory exists
        uvm_path="/sys/kernel/debug/nvidia-uvm/processes/$pid/0/compute.priority"

        if [ ! -f "$uvm_path" ]; then
            echo "Warning: UVM path not found for Diffusion PID $pid: $uvm_path"
            continue
        fi

        # Set the compute priority to $DIFFUSION_PRIORITY for diffusion
        echo "Setting compute priority to $DIFFUSION_PRIORITY for Diffusion PID $pid"
        echo $DIFFUSION_PRIORITY | sudo tee "$uvm_path" > /dev/null

        if [ $? -eq 0 ]; then
            echo "Successfully set compute priority to $DIFFUSION_PRIORITY for Diffusion PID $pid"
        else
            echo "Failed to set compute priority to $DIFFUSION_PRIORITY for Diffusion PID $pid"
        fi
    done
fi
}

set_vllm_priority
echo ""
set_diffusion_priority