#!/bin/bash

# Standalone script to test diffusion model throughput with different memory limits
# This version doesn't require vLLM server and tests diffusion in isolation
# Usage: ./test_diffusion_memory_limits_standalone.sh [gpu_id]

set -Eeuo pipefail

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
source ${SCRIPT_DIR}/../utils/gvm_utils.sh
# Set up GVM environment
source ${SCRIPT_DIR}/../utils/setup_scheduler.sh gvm 0


# Configuration
GPU_ID=${1:-0}
NUM_REQUESTS=5
DATASET_PATH="datasets/vidprom.txt"
RESULTS_DIR="diffusion_tput_wrt_mem"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Memory limits to test (in MB)
MEMORY_LIMITS=(22528 20480 19456 18432 16384 14336 12288 10240 8192 6144 4096 2048)

# Ensure results directory exists
mkdir -p ${RESULTS_DIR}

echo "Starting standalone memory limit testing for diffusion model..."
echo "GPU ID: ${GPU_ID}"
echo "Number of requests per test: ${NUM_REQUESTS}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

# Check if we have the necessary permissions for GVM
if [ ! -w "/sys/kernel/debug/nvidia-uvm/" ]; then
    echo "Warning: No write access to /sys/kernel/debug/nvidia-uvm/"
    echo "Memory limits may not work. Please run with sudo or ensure proper permissions."
    echo ""
fi

# Function to run diffusion test with specific memory limit
run_diffusion_test() {
    local memory_limit_mb=$1
    local test_name="memory_${memory_limit_mb}mb"
    local log_file="${RESULTS_DIR}/${test_name}/log.txt"

    echo "Testing with memory limit: ${memory_limit_mb} MB"
    mkdir -p ${RESULTS_DIR}/${test_name}

    # Start diffusion process
    echo "Starting diffusion process..."
    python diffusion.py \
        --dataset_path ${DATASET_PATH} \
        --num_requests ${NUM_REQUESTS} \
        --log_file stats.txt \
        --output_dir ${RESULTS_DIR}/${test_name} > ${log_file} 2>&1 &

    local diffusion_pid=$!

    # Wait for the process to initialize and start using GPU
    while true; do
        diffusion_pid=$(find_diffusion_pids)
        if [ -n "${diffusion_pid}" ]; then
            break
        fi
        sleep 2
    done

    # Apply memory limit using GVM utilities
    echo "Diffusion process started with PID: ${diffusion_pid} and using memory limit of ${memory_limit_mb} MB"
    set_memory_limit_in_mb ${diffusion_pid} ${memory_limit_mb} ${GPU_ID}

    # Wait for diffusion process to complete
    echo "Waiting for diffusion process to complete..."
    local start_time=$(date +%s)
    wait ${diffusion_pid}
    local exit_code=$?
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))

    echo "Diffusion process completed in ${total_time} seconds (exit code: ${exit_code})"
    echo ""
}

# Cleanup function
cleanup() {
    kill_diffusion
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
echo "=== Starting Standalone Memory Limit Testing ==="

# Run tests for each memory limit
for memory_limit in "${MEMORY_LIMITS[@]}"; do
    echo "----------------------------------------"
    run_diffusion_test ${memory_limit}

    # Small delay between tests to ensure clean separation
    sleep 5
done

echo "=== All tests completed ==="
