#!/usr/bin/env bash

set -euo pipefail

output_path="${1:?usage: $0 OUTPUT_CSV [INTERVAL_SECONDS]}"
interval_seconds="${2:-0.5}"

printf '%s\n' \
  'timestamp_ns,hostname,gpu_index,gpu_uuid,memory_used_mib,memory_total_mib,gpu_util_percent,memory_util_percent' \
  > "$output_path"

while true; do
  timestamp_ns="$(date +%s%N)"
  hostname_value="$(hostname)"
  nvidia-smi \
    --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory \
    --format=csv,noheader,nounits | \
    awk -F ', ' -v timestamp="$timestamp_ns" -v host="$hostname_value" \
      '{printf "%s,%s,%s,%s,%s,%s,%s,%s\n", timestamp, host, $1, $2, $3, $4, $5, $6}' \
      >> "$output_path"
  sleep "$interval_seconds"
done
