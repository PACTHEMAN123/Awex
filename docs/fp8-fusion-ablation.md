# FP8 fusion ablation

`SHARDSTREAM_FP8_EXECUTION=fused` (default) quantizes BF16 source views directly
into the transport FIFO. `staged` first materializes all unique FP8 source views
in a persistent GPU buffer with a separate kernel, then runs the same transport
schedule. The buffer is allocated once during plan preparation, with source
views deduplicated across peers and channel fragments. Its contents are rebuilt
on every publication. Receiver scatter and relay paths are unchanged.

Both modes use the same binary, quantizer, E4M3 block records/scales, BF16
exceptions, routes, GIN backend, channel geometry, compact FIFO and DP source
policy. The staged buffer is ordinary source memory; the transport reads it
into the existing FIFO. It does not require an extra registered network window.
The complete update includes the separate quantizer and transport. Preparation,
allocation, checkpoint verification and generation are excluded from steady
state update time. Preparation never quantizes or advances a version.

The recipe runner accepts `--weight-format fp8 --model-weight-profile
--fp8-execution fused|staged`. Staged execution is rejected for other backends
or BF16. Native metrics expose `fp8_execution_staged`, `fp8_staging_bytes`,
`fp8_quantize_job_count`, `fp8_materialization_time_ms`, and
`transport_device_time_ms`. `kernel_transfer_time_ms` includes both GPU stages;
fused materialization time is zero because quantization is inside transport.
Do not interpret receiver wait time as a separate additive quantization cost.

Validate exact wire equivalence (all four block shapes, strided sources,
unaligned record splits, zero inputs, repeated versions):

```
nvcc --expt-extended-lambda --expt-relaxed-constexpr -std=c++17 -arch=sm_90 -DNCCL_OS_LINUX -Iinclude -I$SHARDSTREAM_NCCL_ROOT/include tests/fp8_staged.cu -o /tmp/fp8_staged
/tmp/fp8_staged
```

Model experiments must additionally pass the independent full checkpoint
FP8/scales/BF16 oracle and generation checks. Use 20 updates per launch,
discard the first 5 and measure the remaining 15. Rerun a fresh fused control
alongside staged; historical FP8/BF16 payload scaling is not a fusion control.

Plain TP2 consumers may need `--rollout-gpu-memory-utilization 0.35` to
reserve enough space for the independent checkpoint oracle. Set the same KV
reservation for both modes. Oracle construction and verification remain
outside update timing; never disable them to obtain a performance point.
