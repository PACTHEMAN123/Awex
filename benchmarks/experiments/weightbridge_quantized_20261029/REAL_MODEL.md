# Real Qwen3 Weight Publication

`model_run.py` runs Qwen3-30B-A3B on the existing FP8 fusion layout: 16
Megatron GPUs (TP2, CP2, EP8, expert TP1), four independent TP2 vLLM engines
on eight rollout GPUs, and the same four Hole hosts and NIC-spread placement.
The native WeightBridge round cap remains 2 GiB, with 128 MiB logical tensors.

`weightbridge_fp8_model.py` discovers the real Megatron correspondence using
native WeightBridge, extracts current parameters through that inferred map,
gathers partial tensors within their native TP group, and quantizes complete
physical HF tensors. Expert TP1 tensors are already complete and need no gather.
Native WeightBridge independently probes the persistent FP8/scale staging
views and actual vLLM loader, then handles routing, packing and RDMA unchanged.
No hand-written LoadSpec or checkpoint read is used inside measured updates.

The actual reference vLLM layout quantizes router weights too: its
`ignored_layers` wildcard is not expanded by this vLLM version. This extension
matches that actual layout, including routers, instead of changing the baseline
configuration. Embedding, head and normalization parameters remain mixed dtype.

This first implementation materializes all local quantized payloads before
native sending, not one tensor group per internal RDMA round. Full-tensor
quantization is per physical tensor, without batching Python launches. It is
an eager prototype, not a claim about the best optimized WeightBridge latency
or a memory-bounded implementation of the proposed extension.

The shared harness publishes actual live model weights with changing norm
version markers; it does not run optimizer steps. One checkpoint bootstrap is
followed by the requested updates. Checkpoint reads, independent full-value
checks and generation checks are outside publication timing. Gather, extraction,
quantization, staging, native transfer and all-receiver consumption are inside.
The reference also uses this publication-only definition of end-to-end.

FP8 bytes are checked using the existing independent FP64 rounding-interval
oracle (four FP32 eps relative bound), not by rerunning the quantizer. Scales,
mixed values, version markers, stable pointers and deterministic generation
must all pass. Do not call ambiguous-midpoint acceptance exact byte equality.

After downloading all four archives and metadata, run `model_analyze.py` with
`--reference-outcome` pointing to the prior fused run's outcome JSON. It checks
archive hashes, actual placement/configuration, complete receiver/version
coverage, preparation records and cleanup before emitting latency summaries.
Logical NCCL gather payload is not a measured NIC counter.
