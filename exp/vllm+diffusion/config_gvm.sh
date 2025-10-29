#!/bin/bash

# Script to find vLLM process using GPU and set compute priority
# Usage: ./config_gvm.sh <VLLM_PRIORITY> <DIFFUSION_PRIORITY>

set -Eeuo pipefail

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
source ${SCRIPT_DIR}/../utils/gvm_utils.sh

VLLM_PRIORITY=${1:-0}
DIFFUSION_PRIORITY=${2:-8}
VLLM_MEM_GB=${3:-0}
DIFFUSION_MEM_GB=${4:-0}

# Generic function to apply an action (priority or memory limit) to processes
set_process_config() {
    local finder_func=$1
    local process_type=$2
    local action=$3
    local value=$4
    
    # Setup action-specific variables
    local action_desc action_func
    if [ "$action" = "priority" ]; then
        action_desc="compute priority"
        action_func="set_compute_priority"
        echo "Looking for $process_type processes using GPU..."
    else
        action_desc="memory limit"
        action_func="set_memory_limit_in_gb"
        echo "Setting memory limit to ${value}GB for $process_type processes using GPU..."
    fi
    
    local pids=$($finder_func)
    
    if [ -z "$pids" ]; then
        echo "No $process_type processes found using GPU"
    else
        echo "Found $process_type processes with PIDs: $pids"
        
        for pid in $pids; do
            # Skip memory limit if value is 0
            if [ "$action" = "priority" ] || [ $value -gt 0 ]; then
                $action_func $pid $value
                if [ $? -eq 0 ]; then
                    echo "Successfully set $action_desc to $value for $process_type PID $pid"
                else
                    echo "Failed to set $action_desc to $value for $process_type PID $pid"
                fi
            fi
        done
        
        [ "$action" = "priority" ] && echo "Done processing all $process_type processes"
    fi
}

init_debugfs
set_process_config find_vllm_pids "vLLM" priority "${VLLM_PRIORITY}"
set_process_config find_vllm_pids "vLLM" memory "${VLLM_MEM_GB}"
echo ""
set_process_config find_diffusion_pids "Diffusion" priority "${DIFFUSION_PRIORITY}"
set_process_config find_diffusion_pids "Diffusion" memory "${DIFFUSION_MEM_GB}"
