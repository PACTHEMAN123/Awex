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
Both block dimensions implement 64 or 128. GPU qualification uses 128×128
blocks and Qwen3-30B TP2 expert width 384. TP4 width-192 qualification is
deferred. Shards must be block-aligned; unsupported shard/physical-span
boundaries fail before launching the kernel.

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
`fp8_transfer_config.fp8_server_args()` supplies the native FP8 configuration
and Triton backend arguments to both existing vLLM weight-exchange harnesses
when the feature is enabled. Use `comm_backend=nccl_device_v2` on the writer
and reader, and export the same feature/block variables on all ranks.

For pure-GIN FP8 direct communicators with a maximum quantizing-source peer
count of two through seven, registration uses network-sized physical FIFO
slots, allowing contiguous puts across existing slots. A collective checks
for non-ring LSA edges before choosing this layout; mixed non-ring LSA keeps
the legacy slot size. Direct FP8 communicators also negotiate a larger
compute-aware channel budget (12 channels per GIN connection rather than 6,
capped by the existing total channel count). This overlaps sender quantization
across more CTAs without increasing FIFO registration, GIN contexts, signals
or connection count. A collective detects source fanout, rather than receiver
fan-in, before choosing this policy. Two or three source peers use 9 channels
per GIN connection with compact physical slots and the existing credit policy.
At least eight source peers retain the established physical slot,
channel-budget and credit policy: the increased compute concurrency regressed
shared-rail FIFO waits in full-model measurements. Quantization remains fused
in each policy.
Ring and copy-only channel allocation remain unchanged.

Development validation uses `awex/tests/experimental/nccl_device_v2_fp8_e2e.py`
on real multi-node GIN/LSA transports: ten changing BF16 updates, exact FP8-byte
comparison against a PyTorch block reference, scale checks and padding guards.
`AWEX_FP8_CHECK_CONTIGUOUS=1` also exercises the contiguous-matrix descriptor.
The full 30B experiment uses the existing weight-exchange harness with ten
updates, discarding the first three. Complete update latency includes fused
quantization, scale transfer, relaying and target writes. Report actual per-pair
wire bytes separately from the original BF16 checkpoint size; receiver payload
metrics include incoming plus forwarded bytes and are not one-way TX volume.

The first complete TP2 matrix measured all three ring modes at fanouts 2, 4
and 8 on four H20 nodes (two training, two inference). FP8 complete-update
speedups were 1.80–1.98× in eight of nine cases. Direct fanout 4 was initially
1.46×; the compute-aware channel policy improved its matched repeat from
1018.138 ms BF16 to 545.383 ms FP8 (1.87×). Swizzle fanout 8 measured
575.642 ms BF16 versus 298.609 ms FP8 in the first matrix, and
586.716 ms versus 313.762 ms in the repeat.

The subsequent complete matrix restores direct fanout 8 to 2160.223 ms BF16
versus 1086.182 ms FP8. Swizzle fanouts 4 and 8 measure 356.312 ms and
289.662 ms FP8, respectively. Direct fanout 8 has a visible latency tail:
FP8 p95 is 1973.412 ms, with all seven measured samples retained. These are
complete-update p50 results, not uniform tail-latency improvements.

Direct fanout 2's latest matched
128 KiB / 16-slot pair is 1068.381 ms BF16 versus 534.686 ms FP8 (1.998×).
The faster historical 938.215 ms BF16 control gives a conservative 1.755×.
Larger repeated BF16 control times are not used to inflate the speedup.
FIFO-step follow-up measurements are archived with their actual settings.

The selected TP2 measurements below use the repaired small direct-fanout
policy and the unchanged medium/high-fanout and ring paths. Source revisions
541a4cd, 292dd22 and 332ee00 retain identical compiled device instructions;
their host-policy difference is confined to small direct FP8 communicators.
All nine pairs retain ten updates, three warmup updates and seven steady
samples. Each BF16/FP8 pair uses the same FIFO configuration.

| Inference | Ring | Step KiB / depth | BF16 p50 ms | FP8 p50 ms | Paired gain |
| --- | --- | ---: | ---: | ---: | ---: |
| TP2×2 | off | 128 / 16 | 1068.381 | 534.686 | 1.998× |
| TP2×2 | naive | 128 / 16 | 1308.588 | 701.250 | 1.866× |
| TP2×2 | swizzle | 128 / 16 | 1140.250 | 597.765 | 1.908× |
| TP2×4 | off | 128 / 16 | 1129.992 | 552.976 | 2.043× |
| TP2×4 | naive | 128 / 16 | 1335.995 | 709.396 | 1.883× |
| TP2×4 | swizzle | 512 / 4 | 640.931 | 332.427 | 1.928× |
| TP2×8 | off | 128 / 16 | 2160.223 | 1086.182 | 1.989× |
| TP2×8 | naive | 128 / 16 | 3380.415 | 1217.359 | 2.777× |
| TP2×8 | swizzle | 128 / 16 | 571.659 | 289.662 | 1.974× |

The naive×8 paired BF16 control is slower than its best archived repeat.
Comparing each selected FP8 case with its fastest archived BF16 control
across FIFO configurations gives 1.755–1.974×, about 43.0–49.3% lower p50.
These controls include the preceding ring experiment when topology, GPU
rails, update counts, runtime limits and logical payload match. The faster
608.893-ms swizzle×4 BF16 control gives 1.832× against 332.427 ms FP8.
All samples, including the direct×8 tail above, remain in the local archive.

To reproduce the selected swizzle×4 FIFO tuning, set the following on every
rank for both BF16 and FP8 runs; other selected cases use 128 KiB / depth 16:

```bash
export AWEX_NCCL_DEVICE_V2_NET_STEP_BYTES=524288
export AWEX_NCCL_DEVICE_V2_GIN_FIFO_DEPTH=4
```

The default FIFO settings remain unchanged. The final device binary passes
ten multi-node exact/legacy variants, including delayed-reader FIFO wrap,
small/medium/high source fanout, GIN/LSA mixtures, naive/swizzle and feature-off
ring/direct transfer. The selected 512-KiB ring configuration additionally
passes exact changing-weight byte/scale checks and full-model BF16/FP8
generation checks. Transfer correctness and generation consistency are
validated; task-level model quality and TP4 width-192 cases are not evaluated.
