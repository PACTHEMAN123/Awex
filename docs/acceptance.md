# Required experiment acceptance

Baseline: Awex `codex/device-v2-dynamic-rollout`, revision
`6dabfbf7ede790c34011924136f132514fa4b60c`. New library: ShardStream.
Historical baseline results are comparison evidence, never new-library passes.

| Experiment | Required coverage | Current new-library evidence |
|---|---|---|
| 8 | Real veRL GRPO BF16, original NCCL controls, six topologies | First TP4/CP2/EP8→inference TP8 case running at `3b69d74`; matrix pending |
| 9 | Real model weight exchange, three fanout topologies, off/naive/swizzle routes with controls | Pending integration and GPU runs |
| 10 | Three fanouts × three routes, paired BF16/FP8, weight bytes/scales/cache/generation | CPU FP8 descriptor regression only; GPU pending |
| 11 | Real veRL FP8, training TP2/TP4 to inference TP2×8, original paired controls | Pending integration and GPU runs |
| 12 | Real Qwen3-30B-A3B BF16, 2→4→8 model instances, full equality, first publication cache hit, background preparation timeline | At `3b69d74`: real-model correctness, cache, background compute and steady transfer comparison pass; preparation timing investigation remains open |

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

## Real Elastic qualification at `3b69d74`

Four user-selected H20 nodes, the same local Qwen3-30B-A3B checkpoint and original
transfer geometry were used for adjacent ShardStream and original `6dabfbf`
runs. Each inference instance contains 61,089,832,960 parameter bytes. Each run
passes 280 full parameter equality checks over 30 publications, with three
warmup publications per cohort. ShardStream raw worker logs contain 760 transfer
records; all 740 after initial preparation hit cache and report zero transport
initialization, plan initialization, metadata upload, host lowering and batch
building time. All 56 worker preparations match the original on 22 resource
configuration metrics, including connections, contexts, channels, FIFO depth,
step sizes and signals.

| Instances | Original steady p50 | ShardStream steady p50 | Difference | First publication after join |
|---|---:|---:|---:|---:|
| 2 | 1161.87 ms | 1128.53 ms | −2.87% | Initial cold publication: 24.55 s |
| 4 | 636.55 ms | 639.93 ms | +0.53% | 642.16 ms |
| 8 | 570.28 ms | 550.95 ms | −3.39% | 556.49 ms |

This single paired run shows no material steady transfer regression; it does
not establish statistical equivalence. Full prepare takes 29.53/26.96 seconds
for ShardStream and 18.81/33.04 seconds for the original. Training rank zero
attributes 18.50/17.30 seconds to GIN device communicator initialization versus
10.63/19.66 seconds in the control. Model plan building takes 0.43/0.63 seconds
versus 0.46/0.65 seconds. These phase timings localize most of the variation to
communicator setup; they do not prove its cause or establish prepare equivalence.

All 20/24 existing workers complete bounded BF16 GEMMs inside native preparation
intervals. The timeline uses worker clocks and CUDA completion events; it proves
overlap with this compute workload, not optimizer steps or token generation.
Local raw logs, profiles, timeline PNG/PDF, Chrome trace and audit are in
`../results/shardstream-refactor-20261009/elastic-bf16-r2/`; adjacent comparison
and phase records are in the parent results directory. No OSS durability is
claimed. The subsequent removal of uncalled helpers still needs a refreshed
wheel and GPU qualification.

Original experimental reports remain outside the library in `../results/`:

- `awex-topology-matrix-20261006/topology-matrix-report.md`
- `awex-ring-microbench-20261007/HIGHER-FANOUT.md`
- `awex-fp8-blockwise-20261007/FINAL-WORKFLOW-IMPACT.md`
- `verl-fp8-topology-20261008/REPORT.md`
- `awex-elastic-prepare-opt-20261009/RESULTS.md`
