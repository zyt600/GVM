[[ ${__GVM_UTILS_SH__:-} ]] && return 0; readonly __GVM_UTILS_SH__=1

# Strict mode for the library (safe in functions)
shopt -s lastpipe
__gvm_err(){ printf 'ERR: %s\n' "$*" >&2; }
die(){ __gvm_err "$@"; exit 1; }


init_debugfs() {
    echo "Initializing GVM debugfs..."
    sudo chmod 777 /sys/kernel/debug/
    sudo chmod -R 777 /sys/kernel/debug/nvidia-uvm/
    echo "Done"
}

find_pids() {
    local pattern=$1
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')

    local result_pids=""
    for gpu_pid in $gpu_pids; do
        if ps -p "$gpu_pid" -o cmd --no-headers 2>/dev/null | grep -iq "$pattern"; then
            result_pids="$result_pids $gpu_pid"
        fi
    done

    # Remove leading/trailing spaces and output
    echo "$result_pids" | xargs
}

set_compute_priority() {
    local pid=$1
    local priority=$2

    if [ ! -d "/sys/kernel/debug/nvidia-uvm/processes/$pid" ]; then
        die "Directory /sys/kernel/debug/nvidia-uvm/processes/$pid does not exist"
    fi

    for dir in /sys/kernel/debug/nvidia-uvm/processes/$pid/*; do
        if [ -d "$dir" ]; then
            if [ -w "$dir/compute.priority" ]; then
                echo "Setting compute priority for PID $pid, dir $dir"
                echo $priority | sudo tee "$dir/compute.priority" > /dev/null
            else
                die "Permission denied to set compute priority for PID $pid, dir $dir"
            fi
        fi
    done
}

set_memory_limit_in_bytes() {
    local pid=$1
    local limit=$2
    local gpuid=${3:-0}

    if [ ! -w "/sys/kernel/debug/nvidia-uvm/processes/$pid/$gpuid/memory.limit" ]; then
        die "Permission denied to set memory limit for PID $pid, gpuid $gpuid"
    fi

    echo $limit | sudo tee "/sys/kernel/debug/nvidia-uvm/processes/$pid/$gpuid/memory.limit" > /dev/null
}

set_memory_limit_in_mb() {
    local pid=$1
    local limit=$2
    local gpuid=${3:-0}

    set_memory_limit_in_bytes $pid $((limit * 1024 * 1024)) $gpuid
}

set_memory_limit_in_gb() {
    local pid=$1
    local limit=$2
    local gpuid=${3:-0}

    set_memory_limit_in_bytes $pid $((limit * 1024 * 1024 * 1024)) $gpuid
}

find_vllm_pids() {
    find_pids ".*vllm.*"
}

find_diffusion_pids() {
    find_pids "python.*diffusion\.py"
}
