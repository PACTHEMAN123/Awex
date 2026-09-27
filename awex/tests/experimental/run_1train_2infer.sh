#!/usr/bin/env bash

set -euo pipefail

role="${1:?usage: $0 train|infer EXPERIMENT BACKEND}"
experiment="${2:?experiment must be B1, B2, B3, or B4}"
backend="${3:?backend must be verl-nccl-bucket, awex-nccl, or awex-nccl-device-v2-gin}"

: "${MODEL_PATH:?set MODEL_PATH}"

base_port="${BASE_PORT:-18000}"
num_updates="${NUM_UPDATES:-7}"
warmup_updates="${WARMUP_UPDATES:-2}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"

case "$experiment" in
  B1)
    train_cp=1 infer_tp=4 engines_per_node=2 total_engines=4 infer_ep=0
    ;;
  B2)
    train_cp=2 infer_tp=4 engines_per_node=2 total_engines=4 infer_ep=0
    ;;
  B3)
    train_cp=1 infer_tp=8 engines_per_node=1 total_engines=2 infer_ep=1
    ;;
  B4)
    train_cp=2 infer_tp=8 engines_per_node=1 total_engines=2 infer_ep=1
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
    --num-engines "$engines_per_node" \
    --vllm-gpu-memory-utilization "$gpu_memory_utilization" \
    --num-updates "$num_updates" \
    --profile \
    --warmup-updates "$warmup_updates" \
    --sync-transfer-start \
    "${expert_parallel_args[@]}"
fi

if [[ "$role" != train ]]; then
  printf 'Unknown role: %s\n' "$role" >&2
  exit 2
fi

: "${TRAIN_DRIVER_HOST:?set TRAIN_DRIVER_HOST to the training container IP}"
: "${INFERENCE_HOST_0:?set INFERENCE_HOST_0 to inference node 0 container IP}"
: "${INFERENCE_HOST_1:?set INFERENCE_HOST_1 to inference node 1 container IP}"

endpoint_args=()
for ((local_rank = 0; local_rank < engines_per_node; local_rank++)); do
  endpoint_args+=(
    --inference-endpoint
    "${local_rank},${INFERENCE_HOST_0},$((base_port + local_rank))"
  )
done
for ((local_rank = 0; local_rank < engines_per_node; local_rank++)); do
  endpoint_args+=(
    --inference-endpoint
    "$((engines_per_node + local_rank)),${INFERENCE_HOST_1},$((base_port + local_rank))"
  )
done

exec python -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node=8 \
  --master-addr="$TRAIN_DRIVER_HOST" \
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
  --train-tp-size 4 \
  --train-pp-size 1 \
  --train-cp-size "$train_cp" \
  --train-ep-size 8 \
  --train-expert-tp-size 1 \
  --vllm-tp-size "$infer_tp" \
  --num-engines "$total_engines" \
  --vllm-gpu-memory-utilization "$gpu_memory_utilization" \
  --use-mbridge \
  --profile \
  --sync-transfer-start \
  --num-updates "$num_updates" \
  --warmup-updates "$warmup_updates" \
  "${expert_parallel_args[@]}"
