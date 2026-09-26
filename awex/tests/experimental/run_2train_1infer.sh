#!/usr/bin/env bash

set -euo pipefail

role="${1:?usage: $0 train|infer EXPERIMENT BACKEND}"
experiment="${2:?experiment must be A1, A2, A3, or A4}"
backend="${3:?backend must be verl-nccl-bucket, awex-nccl, or awex-nccl-device-v2-gin}"

: "${MODEL_PATH:?set MODEL_PATH}"
: "${INFERENCE_HOST:?set INFERENCE_HOST to the inference container IP}"

base_port="${BASE_PORT:-18000}"
num_updates="${NUM_UPDATES:-4}"
warmup_updates="${WARMUP_UPDATES:-1}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"

case "$experiment" in
  A1)
    train_tp=2 train_pp=4 train_cp=2 train_ep=4 train_etp=1
    infer_tp=4 num_engines=2 infer_ep=0
    ;;
  A2)
    train_tp=2 train_pp=4 train_cp=2 train_ep=4 train_etp=1
    infer_tp=8 num_engines=1 infer_ep=1
    ;;
  A3)
    train_tp=4 train_pp=1 train_cp=2 train_ep=8 train_etp=1
    infer_tp=4 num_engines=2 infer_ep=0
    ;;
  A4)
    train_tp=4 train_pp=1 train_cp=2 train_ep=8 train_etp=1
    infer_tp=8 num_engines=1 infer_ep=1
    ;;
  *)
    printf 'Unknown experiment: %s\n' "$experiment" >&2
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

expert_parallel_args=()
if [[ "$infer_ep" == 1 ]]; then
  expert_parallel_args+=(--vllm-enable-expert-parallel)
fi

if [[ "$role" == infer ]]; then
  exec python -m awex.tests.weights_exchange_vllm_infer_it \
    --server-only \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --client-host 127.0.0.1 \
    --port "$base_port" \
    --vllm-tp-size "$infer_tp" \
    --num-engines "$num_engines" \
    --vllm-gpu-memory-utilization "$gpu_memory_utilization" \
    --sync-transfer-start \
    "${expert_parallel_args[@]}"
fi

if [[ "$role" != train ]]; then
  printf 'Unknown role: %s\n' "$role" >&2
  exit 2
fi

: "${MASTER_ADDR:?set MASTER_ADDR to training node-rank 0 container IP}"
: "${NODE_RANK:?set NODE_RANK to 0 or 1}"
: "${TRAIN_DRIVER_HOST:?set TRAIN_DRIVER_HOST to training node-rank 0 container IP}"

exec python -m torch.distributed.run \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="${MASTER_PORT:-17999}" \
  -m awex.tests.weights_exchange_multi_vllm_it \
  --remote-inference \
  --host "$INFERENCE_HOST" \
  --port "$base_port" \
  --meta-server-host "$TRAIN_DRIVER_HOST" \
  --meta-server-port="${META_SERVER_PORT:-17998}" \
  --publication-store-host "$TRAIN_DRIVER_HOST" \
  --publication-mechanism "$publication" \
  --comm_backend "$comm_backend" \
  --model-path "$MODEL_PATH" \
  --train-tp-size "$train_tp" \
  --train-pp-size "$train_pp" \
  --train-cp-size "$train_cp" \
  --train-ep-size "$train_ep" \
  --train-expert-tp-size "$train_etp" \
  --vllm-tp-size "$infer_tp" \
  --num-engines "$num_engines" \
  --vllm-gpu-memory-utilization "$gpu_memory_utilization" \
  --use-mbridge \
  --profile \
  --sync-transfer-start \
  --num-updates "$num_updates" \
  --warmup-updates "$warmup_updates" \
  "${expert_parallel_args[@]}"
