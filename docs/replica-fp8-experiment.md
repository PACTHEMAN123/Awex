# Native FP8 supplement to the replica scaling matrix

Use Qwen3-30B-A3B, the complete logical replica routing / stream source / source-node
ring / NIC spread scheme, and the same four eight-GPU H20-3e nodes as the BF16 matrix.
No feature-disable controls are requested.

Training retains BF16 checkpoint weights and changes to TP2/CP4/EP8/ETP1, PP1,
16 GPUs, dense/expert DP2/2. ShardStream fuses BF16-to-E4M3 quantization into its
existing transport kernel. Native inference FP8 matrices use 128×128 blocks and
FP32 dequantization scales; routers, embeddings, norms and output layers retain
their configured unquantized storage. This is a weight-publication benchmark,
not FP8 training compute or an end-to-end training throughput measurement.

| Inference TP | Inference DP | EP | Instances | GPUs per instance |
|---|---|---|---|---|
| 2 | 4 | 8 | 2 | 8 |
| 1 | 8 | 8 | 2 | 8 |

Run three independent launches per row, each with five warmup and thirty measured
publications. One separately recorded checkpoint bootstrap per launch populates
vLLM's dummy native FP8 storage before taking the deterministic DP0 generation
baseline. It is not a steady sample. Version -1 uses the original checkpoint norm;
the regular 0..34 schedule uses distinct norm factors and restores the norm on its final
publication. Thus six formal launches contain 210 regular verified updates,
180 measured updates, and six additional verified checkpoint bootstraps.

After the repeated generation baseline, drain and pause each instance using
vLLM's `POST /pause?mode=wait` and require its acknowledgement before regular
publications. This prevents DP/EP dummy-batch coordination from interleaving with
weight-update worker RPCs. The benchmark enables vLLM's development endpoints
for this supported weight-update lifecycle. DP, EP and all transport features
remain enabled. Resume each instance before the final generation check. Pause
and resume are outside publication timing. Pre-pause failed attempts and their
debugger evidence are retained separately from the frozen formal matrix.

Outside each timed publication, clear every destination and scale, then check
all canonical parameter views against the real BF16 safetensors checkpoint:
FP8 encoded bytes exactly, scales against an independent block amax reference
(rtol 5e-7, atol 0), and remaining BF16/FP32 parameters exactly. The byte oracle
uses the independently validated transmitted FP32 scale and a reciprocal rounded
through FP64 to FP32, matching scalar CUDA division without treating received
weight bytes as a reference. Cached expected bytes are independently computed;
they are invalidated if the validated scale changes. Check version markers and
stable parameter pointers for every update, including the bootstrap. Compare
uncached generation before and after the final restored checkpoint.

The time interval is `publication.publish()` through device synchronization.
Reference loading, clearing, comparisons, telemetry and generation are excluded.
Record the bootstrap cold publication separately from regular warmups and samples.
Distinguish logical BF16 source coverage, actual wire payload including FP8 block
records, and physical NIC counters. Their byte counts are not interchangeable.

Both source TP and CP differ from the preceding BF16 matrix. Any displayed
comparison to its TP2/TP1 inference points is a comparison of complete
configurations, not a precision-only ablation. This restricted matrix does not
establish a general claim that every FP8 topology with TP4 or greater is invalid.

Preserve all launch outcomes and four-node raw logs, actual GPU/HCA binding,
source ownership and physical-node chains. Freeze one source/native revision
for the formal matrix. Verify scoped cleanup and all 32 GPUs free at completion;
mirror artifacts locally and on the selected head and verify SHA256 manifests.
