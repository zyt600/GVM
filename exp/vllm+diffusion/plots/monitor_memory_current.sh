#!/usr/bin/env bash
# Long-running monitor of process -> memory.current
# Appends one row per PID per second to a CSV file.
#
# Usage:
#   ./monitor_memory_current.sh [OUTPUT_CSV] [PROCESS_NAME_REGEX]
#
# Example:
#   ./monitor_memory_current.sh vllm_mem.csv '.*vllm.*'
#   ./monitor_memory_current.sh all_mem.csv               # all GPU processes

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../utils/gvm_utils.sh"

OUT_CSV="${1:-"./memory_current_log.csv"}"
FILTER_REGEX="${2:-""}"

init_debugfs

# Write CSV header if file doesn't exist
if [ ! -f "$OUT_CSV" ]; then
  echo "timestamp,pid,cmd,gpu_id,memory_current_bytes" > "$OUT_CSV"
fi

_all_gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' | sed '/^$/d' | sort -u
}

while true; do
  TS="$(date -Is)"

  # Candidate PIDs
  if [[ -n "$FILTER_REGEX" ]]; then
    mapfile -t PIDS < <(find_pids "$FILTER_REGEX")
  else
    mapfile -t PIDS < <(_all_gpu_pids)
  fi

  for pid in "${PIDS[@]}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    base_dir="/sys/kernel/debug/nvidia-uvm/processes/${pid}"
    [[ -d "$base_dir" ]] || continue

    cmd="$(ps -p "$pid" -o cmd= 2>/dev/null || echo "<exited>")"

    while IFS= read -r -d '' mcfile; do
      gpu_id="$(basename "$(dirname "$mcfile")")"
      val="$(sudo cat "$mcfile" 2>/dev/null || echo 0)"
      [[ "$val" =~ ^[0-9]+$ ]] || val=0

      printf "%s,%s,%s,%s,%s\n" "$TS" "$pid" "$(echo "$cmd" | tr ',' ' ')" "$gpu_id" "$val" >> "$OUT_CSV"
    done < <(find "$base_dir" -maxdepth 2 -type f -name "memory.current" -print0 2>/dev/null)

  done

  sleep 1
done
