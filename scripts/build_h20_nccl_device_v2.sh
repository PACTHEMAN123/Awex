#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_SOURCE=${1:?usage: build_h20_nccl_device_v2.sh ARCHIVE_SOURCE [TARGET_SOURCE]}
TARGET_SOURCE=${2:-/mnt/fuse/verl-e2e/src/Awex-baseline-adapt}
ROOT=${ROOT:-/mnt/fuse/verl-e2e}
VENV=${VENV:-$ROOT/envs/verl-py312-torch213-cu132-vllm027-pilot}
NCCL_ROOT=${NCCL_ROOT:-/usr/local/cuda}
TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$ROOT/build-cache/torch-extensions-gin-rail-lsa}

mkdir -p "$TARGET_SOURCE" "$TORCH_EXTENSIONS_DIR"
/bin/cp -a "$ARCHIVE_SOURCE"/. "$TARGET_SOURCE"/

cd "$TARGET_SOURCE"
env -u AWEX_NCCL_DEVICE_V2_EXTENSION \
  PYTHONPATH="$TARGET_SOURCE" \
  AWEX_NCCL_INCLUDE="$NCCL_ROOT/include" \
  AWEX_NCCL_LIB="$NCCL_ROOT/lib64" \
  LD_LIBRARY_PATH="$NCCL_ROOT/lib64:${LD_LIBRARY_PATH:-}" \
  LD_PRELOAD="$NCCL_ROOT/lib64/libnccl.so.2" \
  TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
  "$VENV/bin/python" -c \
    'from awex.transfer.nccl_device_v2 import _load_extension; print(_load_extension().__file__)'

/bin/cp -f \
  "$TORCH_EXTENSIONS_DIR/awex_nccl_device_ext_v2/awex_nccl_device_ext_v2.so" \
  "$TARGET_SOURCE/awex_nccl_device_ext_v2.so"
sha256sum "$TARGET_SOURCE/awex_nccl_device_ext_v2.so"
