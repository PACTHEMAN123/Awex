#!/usr/bin/env bash

set -euo pipefail

role="${1:?usage: $0 train|infer FANOUT BACKEND}"
fanout="${2:?fanout must be 1, 2, or 4}"
backend="${3:?backend must be verl-nccl-bucket, awex-nccl, or awex-nccl-device-v2-gin}"

: "${MODEL_PATH:?set MODEL_PATH}"

base_port="${BASE_PORT:-18000}"
num_updates="${NUM_UPDATES:-10}"
warmup_updates="${WARMUP_UPDATES:-3}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
infer_tp=4
ring_mode="${RING_MODE:-off}"

case "$fanout" in
  1|2|4) ;;
  *)
    printf 'Invalid fanout: %s (expected 1, 2, or 4)\n' "$fanout" >&2
    exit 2
    ;;
esac

case "$ring_mode" in
  off)
    export AWEX_NCCL_DEVICE_V2_RING_BROADCAST=0
    export AWEX_NCCL_DEVICE_V2_RING_SWIZZLE=0
    ;;
  naive)
    export AWEX_NCCL_DEVICE_V2_RING_BROADCAST=1
    export AWEX_NCCL_DEVICE_V2_RING_SWIZZLE=0
    ;;
  swizzle)
    export AWEX_NCCL_DEVICE_V2_RING_BROADCAST=1
    export AWEX_NCCL_DEVICE_V2_RING_SWIZZLE=1
    ;;
  *)
    printf 'Invalid RING_MODE: %s (expected off, naive, or swizzle)\n' "$ring_mode" >&2
    exit 2
    ;;
esac

case "$backend" in
  verl-nccl-bucket)
    publication=verl_nccl_broadcast
    comm_backend=nccl
    ;;
  awex-nccl)
    publication=awex
    comm_backend=nccl
    ;;
  awex-nccl-device-v2-gin)
    publication=awex
    comm_backend=nccl_device_v2
    export AWEX_NCCL_DEVICE_V2_HCA_POLICY="${AWEX_NCCL_DEVICE_V2_HCA_POLICY:-balanced}"
    export AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS="${AWEX_NCCL_DEVICE_V2_GIN_CONNECTIONS:-4}"
    export AWEX_NCCL_DEVICE_V2_GIN_FIFO_DEPTH="${AWEX_NCCL_DEVICE_V2_GIN_FIFO_DEPTH:-16}"
    export AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES="${AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES:-131072}"
    ;;
  *)
    printf 'Unknown backend: %s\n' "$backend" >&2
    exit 2
    ;;
esac

# The integration harness launches TP4 engines on eight-GPU inference nodes.
# Keep the logical placement identical to veRL's round-robin engine mapping.
export AWEX_NODE_LOCAL_WORLD_SIZE="${AWEX_NODE_LOCAL_WORLD_SIZE:-8}"

printf 'role=%s fanout=%s backend=%s ring_mode=%s updates=%s warmup=%s\n' \
  "$role" "$fanout" "$backend" "$ring_mode" "$num_updates" "$warmup_updates"

if [[ "$role" == infer ]]; then
  : "${INFERENCE_NODE_INDEX:?set INFERENCE_NODE_INDEX to 0 or 1}"
  case "$INFERENCE_NODE_INDEX" in
    0)
      local_engines=$((fanout == 4 ? 2 : 1))
      ;;
    1)
      if [[ "$fanout" == 1 ]]; then
        printf 'fanout=1 uses only inference node 0\n' >&2
        exit 2
      fi
      local_engines=$((fanout == 4 ? 2 : 1))
      ;;
    *)
      printf 'Invalid INFERENCE_NODE_INDEX: %s\n' "$INFERENCE_NODE_INDEX" >&2
      exit 2
      ;;
  esac

  exec python -m awex.tests.weights_exchange_vllm_infer_it \
    --server-only \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --client-host 127.0.0.1 \
    --port "$base_port" \
    --vllm-tp-size "$infer_tp" \
    --num-engines "$local_engines" \
    --vllm-gpu-memory-utilization "$gpu_memory_utilization" \
    --num-updates "$num_updates" \
    --profile \
    --warmup-updates "$warmup_updates" \
    --sync-transfer-start
fi

if [[ "$role" != train ]]; then
  printf 'Unknown role: %s\n' "$role" >&2
  exit 2
fi

: "${MASTER_ADDR:?set MASTER_ADDR to training node-rank 0 container IP}"
: "${NODE_RANK:?set NODE_RANK to 0 or 1}"
: "${TRAIN_DRIVER_HOST:?set TRAIN_DRIVER_HOST to training node-rank 0 container IP}"
: "${INFERENCE_HOST_0:?set INFERENCE_HOST_0 to inference node 0 container IP}"
: "${INFERENCE_HOST_1:?set INFERENCE_HOST_1 to inference node 1 container IP}"

endpoint_args=(--inference-endpoint "0,${INFERENCE_HOST_0},${base_port}")
if ((fanout >= 2)); then
  endpoint_args+=(--inference-endpoint "1,${INFERENCE_HOST_1},${base_port}")
fi
if ((fanout == 4)); then
  endpoint_args+=(
    --inference-endpoint "2,${INFERENCE_HOST_0},$((base_port + 1))"
    --inference-endpoint "3,${INFERENCE_HOST_1},$((base_port + 1))"
  )
fi

exec python -m torch.distributed.run \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="${MASTER_PORT:-17999}" \
  -m awex.tests.weights_exchange_multi_vllm_it \
  --remote-inference \
  --host "$INFERENCE_HOST_0" \
  --port "$base_port" \
  "${endpoint_args[@]}" \
  --meta-server-host "$TRAIN_DRIVER_HOST" \
  --meta-server-port="${META_SERVER_PORT:-17998}" \
  --publication-store-host "$TRAIN_DRIVER_HOST" \
  --publication-mechanism "$publication" \
  --comm_backend "$comm_backend" \
  --model-path "$MODEL_PATH" \
  --train-tp-size 2 \
  --train-pp-size 1 \
  --train-cp-size 2 \
  --train-ep-size 8 \
  --train-expert-tp-size 1 \
  --vllm-tp-size "$infer_tp" \
  --num-engines "$fanout" \
  --vllm-gpu-memory-utilization "$gpu_memory_utilization" \
  --use-mbridge \
  --profile \
  --sync-transfer-start \
  --num-updates "$num_updates" \
  --warmup-updates "$warmup_updates"
