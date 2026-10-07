# Fused BF16-to-FP8 device-v2 weight transfer

Experimental branch: `codex/device-v2-fp8-blockwise`, based on `c519fff`.

Enable independently of the ring switches:

```bash
export AWEX_NCCL_DEVICE_V2_FP8_BLOCKWISE=1
export AWEX_NCCL_DEVICE_V2_FP8_BLOCK_ROWS=128
export AWEX_NCCL_DEVICE_V2_FP8_BLOCK_COLS=128
export AWEX_NCCL_DEVICE_V2_RING_BROADCAST=1
export AWEX_NCCL_DEVICE_V2_RING_SWIZZLE=1
```

All ranks must agree on the feature and block shape. The default is disabled;
the copy-only API, FIFO credits, GIN coalescing, and ring routes are retained.
Both block dimensions support 64 or 128. Shards must be block-aligned: Qwen3
30B TP2 expert width 384 permits 128; TP4 width 192 requires 64. Unsupported
shard/physical-span boundaries fail before launching the kernel.

The sender remains BF16. A copy-worker warp reads one matrix block, reduces
its absolute maximum with warp shuffles, and encodes finite E4M3 values directly
into the existing FIFO. It does not allocate a quantized model or launch a
separate quantization kernel. The network warp can transmit completed FIFO
steps while copy workers quantize subsequent blocks. Only the original FIFO
publication/credit synchronization is used; quantization adds no CTA barrier.

Each tile record contains a 16-byte header (FP32 scale and 12 zero bytes) and
`block_rows * block_cols` FP8 bytes. Scale is `max(amax, 1e-12) / 448`; encoding
uses reciprocal multiplication and round-to-nearest finite E4M3 conversion.
The receiver writes row-major native FP8 weights and FP32 `weight_scale_inv`
directly. Ring relays forward the identical record, including its scale, before
returning the existing FIFO credit. There is no extra full receive buffer.
Nonquantized parameters, such as embeddings, norms and router weights, retain
their existing dtype and copy path. Quantization is selected from the actual
receiver metadata, not from a blanket cast of every parameter.

The integration harness uses vLLM's native block-FP8 parameter storage with
Triton linear/MoE compute to preserve the exposed row-major weight/scale layout.
The target initializes with dummy loading; actual BF16 checkpoint weights are
then sent from the training model. Generation must occur after the first
successful update. The shared BF16 checkpoint configuration is not modified.

Development validation uses `awex/tests/experimental/nccl_device_v2_fp8_e2e.py`
on real multi-node GIN/LSA transports: ten changing BF16 updates, exact FP8-byte
comparison against a PyTorch block reference, scale checks and padding guards.
`AWEX_FP8_CHECK_CONTIGUOUS=1` also exercises the contiguous-matrix descriptor.
The full 30B experiment uses the existing weight-exchange harness with ten
updates, discarding the first three. Complete update latency includes fused
quantization, scale transfer, relaying and target writes. Report actual per-pair
wire bytes separately from the original BF16 checkpoint size; receiver payload
metrics include incoming plus forwarded bytes and are not one-way TX volume.

Performance acceptance is pending GPU measurements; implementation and small
correctness tests alone do not establish the requested approximately 2x gain.
