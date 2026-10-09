# Required experiment acceptance

Baseline: Awex `codex/device-v2-dynamic-rollout`, revision
`6dabfbf7ede790c34011924136f132514fa4b60c`. New library: ShardStream.
Historical baseline results are comparison evidence, never new-library passes.

| Experiment | Required coverage | Current new-library evidence |
|---|---|---|
| 8 | Real veRL GRPO BF16, original NCCL controls, six topologies | Pending integration and GPU runs |
| 9 | Real model weight exchange, three fanout topologies, off/naive/swizzle routes with controls | Pending integration and GPU runs |
| 10 | Three fanouts × three routes, paired BF16/FP8, weight bytes/scales/cache/generation | CPU FP8 descriptor regression only; GPU pending |
| 11 | Real veRL FP8, training TP2/TP4 to inference TP2×8, original paired controls | Pending integration and GPU runs |
| 12 | Real Qwen3-30B-A3B BF16, 2→4→8 model instances, full equality, first publication cache hit, background preparation timeline | CPU membership/cache regression only; GPU pending |

For each GPU case preserve model/runtime, hardware placement, HCA policy,
tensor layouts, queue/context/channel count, FIFO depths, step/chunk sizes,
signals and compiler settings. Record source revision, extension hash, actual
model bytes, all parameter checks, warmup policy, individual timings and cache
metrics. Run an adjacent original-code control to distinguish environmental
variation from a code regression. Preserve raw profiles and logs and report any
regression before accepting the refactor.

The baseline's complete Elastic prepare is 18.44/21.01 seconds, and the adjacent
eight-instance publication control is about 599 ms. The earlier 10-second
optimization target remains unachieved; it is not a prerequisite to count the
refactor as faithful to baseline behavior.

Original experimental reports remain outside the library in `../results/`:

- `awex-topology-matrix-20261006/topology-matrix-report.md`
- `awex-ring-microbench-20261007/HIGHER-FANOUT.md`
- `awex-fp8-blockwise-20261007/FINAL-WORKFLOW-IMPACT.md`
- `verl-fp8-topology-20261008/REPORT.md`
- `awex-elastic-prepare-opt-20261009/RESULTS.md`
