#!/usr/bin/env bash

set -euo pipefail

role="${1:?usage: $0 train|infer [INFERENCE_NODE_INDEX]}"
inference_node_index="${2:-}"

: "${MODEL_PATH:?set MODEL_PATH}"

log_dir="${LOG_DIR:-../logs/week4-b-w2m5}"
base_port_start="${BASE_PORT_START:-19000}"
num_updates="${NUM_UPDATES:-7}"
warmup_updates="${WARMUP_UPDATES:-2}"
health_timeout_seconds="${HEALTH_TIMEOUT_SECONDS:-900}"

read -r -a experiments <<< "${EXPERIMENTS:-B1 B3 B2 B4}"
read -r -a backends <<< \
  "${BACKENDS:-verl-nccl-bucket awex-nccl awex-nccl-device-v2-gin}"

mkdir -p "$log_dir"

sampler_pid=""

stop_sampler() {
  if [[ -n "$sampler_pid" ]] && kill -0 "$sampler_pid" 2>/dev/null; then
    kill "$sampler_pid"
    wait "$sampler_pid" 2>/dev/null || true
  fi
  sampler_pid=""
}

trap stop_sampler EXIT INT TERM

start_sampler() {
  local output_path="$1"
  awex/tests/experimental/sample_gpu_metrics.sh "$output_path" 0.5 \
    > "${output_path%.csv}-sampler.log" 2>&1 &
  sampler_pid="$!"
}

wait_for_health() {
  local host="$1"
  local port="$2"
  local deadline=$((SECONDS + health_timeout_seconds))
  while (( SECONDS < deadline )); do
    if curl --connect-timeout 1 --max-time 2 --fail --silent \
      "http://${host}:${port}/health" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  printf 'Timed out waiting for %s:%s/health\n' "$host" "$port" >&2
  return 1
}

measure_count() {
  local log_path="$1"
  python - "$log_path" <<'PY'
import json
import sys
from pathlib import Path

count = 0
for line in Path(sys.argv[1]).read_text(errors="replace").splitlines():
    marker = "AWEX_PROFILE "
    if marker not in line:
        continue
    try:
        record = json.loads(line.split(marker, 1)[1])
    except json.JSONDecodeError:
        continue
    if record.get("event") == "end_to_end_update" and record.get("phase") == "measure":
        count += 1
print(count)
PY
}

if [[ "$role" == infer ]]; then
  if [[ "$inference_node_index" != 0 && "$inference_node_index" != 1 ]]; then
    printf 'Inference role requires node index 0 or 1.\n' >&2
    exit 2
  fi
elif [[ "$role" == train ]]; then
  : "${TRAIN_DRIVER_HOST:?set TRAIN_DRIVER_HOST}"
  : "${INFERENCE_HOST_0:?set INFERENCE_HOST_0}"
  : "${INFERENCE_HOST_1:?set INFERENCE_HOST_1}"
else
  printf 'Unknown role: %s\n' "$role" >&2
  exit 2
fi

run_index=0
for experiment in "${experiments[@]}"; do
  case "$experiment" in
    B1|B2) engines_per_node=2 ;;
    B3|B4) engines_per_node=1 ;;
    *)
      printf 'Unknown experiment: %s\n' "$experiment" >&2
      exit 2
      ;;
  esac

  for backend in "${backends[@]}"; do
    prefix="${experiment}-${backend}"
    base_port=$((base_port_start + run_index * 10))
    printf 'Starting %s role=%s base_port=%s\n' "$prefix" "$role" "$base_port"

    if [[ "$role" == infer ]]; then
      node_role="infer-${inference_node_index}"
      log_path="${log_dir}/${prefix}-${node_role}.log"
      start_sampler "${log_dir}/${prefix}-${node_role}-gpu.csv"
      set +e
      MODEL_PATH="$MODEL_PATH" \
      BASE_PORT="$base_port" \
      NUM_UPDATES="$num_updates" \
      WARMUP_UPDATES="$warmup_updates" \
        awex/tests/experimental/run_1train_2infer.sh \
          infer "$experiment" "$backend" 2>&1 | tee "$log_path"
      run_status="${PIPESTATUS[0]}"
      set -e
      stop_sampler
      if [[ "$run_status" -ne 0 ]]; then
        printf '%s inference node %s failed with status %s.\n' \
          "$prefix" "$inference_node_index" "$run_status" >&2
        exit "$run_status"
      fi
    else
      for ((engine = 0; engine < engines_per_node; engine++)); do
        wait_for_health "$INFERENCE_HOST_0" "$((base_port + engine))"
        wait_for_health "$INFERENCE_HOST_1" "$((base_port + engine))"
      done

      log_path="${log_dir}/${prefix}-train.log"
      start_sampler "${log_dir}/${prefix}-train-gpu.csv"
      set +e
      MODEL_PATH="$MODEL_PATH" \
      BASE_PORT="$base_port" \
      NUM_UPDATES="$num_updates" \
      WARMUP_UPDATES="$warmup_updates" \
      TRAIN_DRIVER_HOST="$TRAIN_DRIVER_HOST" \
      INFERENCE_HOST_0="$INFERENCE_HOST_0" \
      INFERENCE_HOST_1="$INFERENCE_HOST_1" \
      MASTER_PORT="$((base_port + 8))" \
      META_SERVER_PORT="$((base_port + 9))" \
        awex/tests/experimental/run_1train_2infer.sh \
          train "$experiment" "$backend" 2>&1 | tee "$log_path"
      run_status="${PIPESTATUS[0]}"
      set -e
      stop_sampler

      completed_measures="$(measure_count "$log_path")"
      if [[ "$completed_measures" -ne 5 ]]; then
        printf '%s produced %s measure records; expected 5 (status %s).\n' \
          "$prefix" "$completed_measures" "$run_status" >&2
        exit 1
      fi
      if [[ "$run_status" -ne 0 ]]; then
        printf '%s completed all measures with teardown status %s.\n' \
          "$prefix" "$run_status" >&2
      fi
    fi

    printf 'Finished %s role=%s\n' "$prefix" "$role"
    run_index=$((run_index + 1))
  done
done
