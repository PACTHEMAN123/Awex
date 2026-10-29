# WeightBridge Gather-Before-Quantize Prototype

This is our experimental extension, not an upstream WeightBridge feature.
It implements pre-transfer quantization of **current trainer weights**:

1. NCCL all-gather the BF16 TP shards to a complete physical tensor on each trainer.
2. Quantize that full tensor with the existing standalone veRL Triton E4M3FN
   128x128-block quantizer.
3. Retain the original TP partition of FP8 values and FP32 dequantization scales.
4. Use unchanged WeightBridge correspondence extraction, routing, packing,
   one-sided RDMA and receiver consumption for those payloads. FP8 storage is
   exposed as uint8 views to preserve its bits, not converted numerically.

The BF16 and quantized loaders are independently probed by native WeightBridge.
No ShardStream transport, manually supplied LoadSpec, or checkpoint read is used
in the timed update. Each update changes the values and scales. Both receivers
check every received element and persistent destination pointer outside timing.

## Supported Scope

- Uniform, block-aligned row/column partitions of contiguous 2-D BF16 matrices.
- Two trainer ranks on two hosts, two independent full-tensor receivers on two
  other hosts in this initial four-node qualification. One GPU per host.
- All physical tensors are prepared before `send_weights()`. Preparation is
  not overlapped with native RDMA rounds. This is an eager pre-publication
  implementation, not a memory-bounded per-native-round implementation.
- All trainers gather and quantize, including redundant quantization of the
  complete tensor. An owner-only gather/quantize/scatter policy is not evaluated.
- For these block-aligned partitions, blockwise FP8 can also be computed locally.
  Full-tensor gathering deliberately follows the quoted extension proposal; it
  is not a mathematical requirement of blockwise FP8 or a lower bound on the
  best achievable quantized WeightBridge implementation.
- No arbitrary Megatron TP/EP/DP/PP layouts, mixed-format model exceptions,
  framework FP8 parameter replacement, or vLLM generation integration yet.
- Consequently these results are microbenchmarks, **not Qwen3-30B end-to-end
  results and not replacement Figure 9/10/13 paper points**.

## Measurements

`bf16`: original native BF16 publication.

`gather_fp8`: gather + full-tensor quantization + staging + native RDMA + wait for
all receivers to consume. Phase timings synchronize GPU work and therefore
include host launch/wait overhead. No overlap is claimed.

`prepared_fp8`: the exact same preparation and changing payload, but preparation
is excluded from the timer. This is a **transfer-only lower bound**, not a valid
end-to-end quantized publication result.

An update's completion latency is the maximum sender duration. Independent
launches are repeated with reversed mode order on odd repetitions. Warmups are
excluded by fixed index; no outlier removal. Correctness checks and the full
participant drain barrier are outside timing.

`logical_gather_egress_bytes` is useful collective payload accounting:
`BF16_tensor_bytes * (trainer_count - 1)` summed over all trainers. It excludes
NCCL protocol headers, control traffic and retransmission, and is **not a NIC
counter measurement**. Native trainer-to-rollout wire bytes come from the actual
WeightBridge route overlaps. Receiver-to-receiver replication is separate.

`peak_allocated_bytes` uses PyTorch allocator statistics, not complete device
residency or every native library allocation. Do not call it total communication
memory.

## Provenance

- Native WeightBridge revision: `17ea1bc6370424d52152880834a94659db5002ec`
  (existing CUDA 13 IPC compatibility patch).
- Default native receiver round cap: 2 GiB; logical tensor cap: 128 MiB.
- Standalone veRL quantizer SHA256:
  `bc04247611fcbb9a0943d6e31bca6d1d765a6b1e2cf82f44bf4d3748a368dd10`.
  Deployment copies its original source unchanged with copyright/license header.
- Original source is local and committed. The deployment helper only creates a
  new experiment checkout and verifies existing native WeightBridge file hashes.
- All four hosts are reserved through the existing campaign lock. Only this
  experiment's actors and coordinator subprocesses are stopped at cleanup.
- Results must be downloaded locally: `/mnt/fuse/verl-e2e` is node-local storage,
  not a durable shared NAS mount.

The deterministic test matrices deliberately have exact power-of-two block
scales, allowing an independent PyTorch full-value oracle with bit-exact FP8
comparison. This validates changing values, partition reconstruction, scale
routing, transport, and in-place consumption; it is not a quantizer accuracy
study on real model weights.
