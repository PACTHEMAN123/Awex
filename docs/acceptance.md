# Required experiment acceptance

Baseline: Awex `codex/device-v2-dynamic-rollout`, revision
`6dabfbf7ede790c34011924136f132514fa4b60c`. New library: ShardStream.
Historical baseline results are comparison evidence, never new-library passes.

| Experiment | Required coverage | Current new-library evidence |
|---|---|---|
| 8 | Real veRL GRPO BF16, historical NCCL controls, main plus three PP1 topologies; PP4 excluded by user | At `3dd7325`: main TP2/CP2/EP8→TP4×4 (40 steps), TP4/CP2/EP8→TP2×8 (40 steps), TP4/CP2/EP8→TP4×4 (40 steps), and TP4/CP2/EP8→TP8×2 (10 steps) pass; BF16 coverage complete |
| 9 | Real model weight exchange, three fanout topologies, off/naive/swizzle routes with controls | All nine required BF16 topology/route cases pass full parameters, cache and generation; TP4×4/naive and TP2×8/off current original/candidate controls show no basic regression; slow historical/main-matrix samples remain preserved, between-run variation cause unproven |
| 10 | Three fanouts × three routes, paired BF16/FP8, weight bytes/scales/cache/generation | All nine BF16 and nine FP8 cases pass full checkpoint checks, cache and generation; FP8 TP2×8/swizzle is within +2.3%/+2.4% of current original; naive performance remains under diagnosis after the final package repeats at 1968.28/2329.06 ms despite the earlier binding prototype's 1283.32/1325.96 ms |
| 11 | Real veRL FP8, training TP2/TP4 to inference TP2×8, historical paired controls | Both TP2/TP4 complete 10 steps, each with 256 cached worker transfers: full updates 314.19/323.15 and 309.85/315.17 ms; historical comparisons recorded |
| 12 | Real Qwen3-30B-A3B BF16, 2→4→8 model instances, full equality, first publication cache hit, background preparation timeline | At `3b69d74`: real-model correctness, cache, background compute and steady transfer comparison pass; prepare timing recorded separately |

For each GPU case preserve model/runtime, hardware placement, HCA policy,
tensor layouts, queue/context/channel count, FIFO depths, step/chunk sizes,
signals and compiler settings. Record source revision, extension hash, actual
model bytes, all parameter checks, warmup policy, individual timings and cache
metrics. Reuse historical original controls with explicit provenance, as requested by
the user. Run a fresh original control only when an actual anomaly requires
diagnosis. Preserve raw profiles and logs, all slow samples, actual full
publication timings and accuracy checks. Preparation latency is recorded but
is not a completion gate.

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
claimed. The subsequent removal of uncalled helpers is installed at `3dd7325` on all
four nodes with an unchanged native extension hash. Its four required BF16
GRPO configurations and both FP8 configurations pass. The standalone model
matrix passes all 21 full-checkpoint, cache and generation checks. The FP8
TP2×8 naive update-latency anomaly remains under diagnosis. Restoring original
GIL ownership produced one near-baseline prototype sample, but the final package
repeats slower. All samples are retained; no transfer resource settings changed.

Original experimental reports remain outside the library in `../results/`:

- `awex-topology-matrix-20261006/topology-matrix-report.md`
- `awex-ring-microbench-20261007/HIGHER-FANOUT.md`
- `awex-fp8-blockwise-20261007/FINAL-WORKFLOW-IMPACT.md`
- `verl-fp8-topology-20261008/REPORT.md`
- `awex-elastic-prepare-opt-20261009/RESULTS.md`

## Real GRPO qualification at `3dd7325`

| Configuration | Publications / complete cycles | Full update p50 / p95 | Trainer step p50 / p95 | Historical update p50 / p95 | Historical step p50 / p95 |
|---|---:|---:|---:|---:|---:|
| TP2/PP1/CP2/EP8/ETP1 FSDP → TP4×4 | 33 / 32 | 597.94 / 652.35 ms | 18.57 / 23.51 s | 1223.50 / 1473.44 ms | 19.45 / 23.90 s |
| TP4/PP1/CP2/EP8/ETP1 FSDP → TP2×8 | 33 / 32 | 543.27 / 550.86 ms | 22.62 / 26.93 s | 2175.63 / 2352.57 ms | 24.49 / 27.82 s |
| TP4/PP1/CP2/EP8/ETP1 FSDP → TP4×4 | 33 / 32 | 618.74 / 647.84 ms | 22.63 / 26.02 s | 1301.19 / 1582.86 ms | 22.70 / 26.32 s |
| TP4/PP1/CP2/EP8/ETP1 FSDP → TP8×2 | 8 / 7 | 660.77 / 695.58 ms | 35.48 / 41.74 s | 764.45 / 768.99 ms | 34.10 / 40.93 s |

The three 40-step cases use warmup versions 0–7 and each passes 1,056
steady cached worker transfers with zero initialization, plan building,
lowering, upload and batch building. The TP8 case completes 10 steps with
warmup versions 0–2 and passes the same checks for 256 steady transfers.
All four actual training processes exit 0. This historical comparison shows
improved publication latency; the TP8 case retains +4.03%/+1.99% trainer-step
p50/p95 variation. It does not attribute changes to the refactor or establish
statistical equivalence. No routine adjacent original rerun is required.

Sources: `../results/awex-weightrail-vs-verl-nccl-h20-20261006-summary.csv`,
`../results/awex-topology-matrix-20261006/topology-matrix-summary.csv`, and
`../results/shardstream-refactor-20261009/historical-grpo-comparison.json`.
All six required PP1 GRPO raw archives are local and SHA-256 verified, as are
all 21 standalone model cases. Correctness coverage is complete; the binding
correction is qualified on real FP8 weights. Final packaged-build correctness
and cache checks also pass, but naive FP8 performance remains unresolved.

The initial naive FP8 standalone update p50/p95 was 2490.84/2760.59 ms.
A fresh candidate recheck reports 1502.32/1839.14 ms, versus the current original
1182.94/1267.27 ms (+27.00%/+45.13%). Both candidate samples remain recorded;
the recheck does not pass the performance gate. The original and candidate
transfer kernels have identical 27,320 SASS instructions, registers, stack and
shared-memory usage. All shared non-time transfer profile fields match except
the backend name, including NIC selection, schedule counts and payload/resource
geometry. These checks exclude those differences; they do not prove the cause
of the latency anomaly. A separate same-process native diagnostic executes both
unchanged backends on each real HF payload, with alternating order. Its combined
publication timing is not comparable to a single-backend update. Its full
checkpoint and cache checks pass, but its timing is excluded from acceptance.
A later audit also finds eight inference HCA bindings differ from the normal
recipe because the diagnostic eagerly imported transport. The contributions
of that binding error and the two communicators are not isolated.

The isolated single-backend binding correction restores original steady
kernel-call GIL ownership, while explicit prepare still releases GIL. It passes
all FP8 bytes, scale, mixed BF16, generation and cache checks. Full-update p50/p95
is 1283.32/1325.96 ms (+8.49%/+4.63% against current original; +5.42%/−20.95%
against historical naive FP8). All seven samples are retained, with population
standard deviation 41.09 ms versus 48.01 ms for the current original and
199.54 ms for the preceding candidate recheck. All 224 rank/step pairs match on
68 shared non-time fields except the backend label. The compiled transfer
kernel is unchanged. This is consistent with the user's accepted roughly 9%
run-to-run difference, not a claim of statistical equivalence or proof that GIL
ownership explains every historical slow sample. The source preserves this
binding correction; C++ runtime, kernel, plan lowering and resource geometry
remain unchanged. The raw archive is locally SHA-256 verified and independently
audited under `../results/shardstream-refactor-20261009/binding-policy-diagnostics/`.

The final `fbce854` package has the same executable binding tokens as that
prototype, and identical transfer SASS. Nevertheless its naive FP8 repeat is
1968.28/2329.06 ms, so the prototype does not resolve the performance gate or
prove GIL ownership caused the earlier anomaly. All 224 rank/step pairs have
identical non-time fields between the prototype and final package. Both pass
full HF/scale/BF16/generation/cache checks. Final BF16 TP4×4 swizzle is
618.56/645.57 ms against historical original 584.55/657.44 ms (+5.82%/−1.81%).
Final FP8 GRPO TP2→TP2×8 updates are 321.74/327.75 ms versus preceding candidate
314.19/323.15 ms (+2.40%/+1.42%), with all 256 cached transfers and 23 resource
fields matched. Trainer-step p50/p95 is 21.74/25.37 s versus 21.66/23.55 s
(+0.37%/+7.73%). Only the independent naive FP8 anomaly remains under diagnosis.

A fresh original `6dabfbf` control after the final package completes all exact
HF byte/scale/mixed-BF16, generation and 224 cache checks at 1291.74/1472.39 ms.
The final package is +52.37%/+58.18% slower; this remains a failed performance gate.
Its 224 rank/step records match the control on 68 shared non-time fields.
Both raw archives are locally SHA-256 verified and independently audited.
Evidence: `../results/shardstream-refactor-20261009/final-naive-vs-fresh-original.json`.

An external single-native isolation attempt eagerly imported transport during
Python startup. Eight inference ranks consequently inherited the wrong HCA
binding and communication timed out. That failed diagnostic is preserved and
excluded from performance/correctness acceptance; it does not explain the
final-package anomaly. The corrected diagnostic defers replacement until the
normal transport import, preserving the existing HCA allocation policy.

## Standalone C++ library qualification

The installed `3dd7325` static archive is consumable through the exported
`ShardStream::shardstream` CMake target. A separate C++ project includes the
public runtime header, forces linkage of create/destroy/launch symbols and runs
without allocating GPUs. Symbol inspection confirms no ATen, c10 or Python
references in the archive, and its executable has no PyTorch/Python dynamic
dependencies. This verifies the native-library packaging requirement; actual
GPU transport evidence comes from the real-model cases above.

Archive SHA-256: `46650f50259869188920c66f80233fb153d14184721a22189b23f7c16ed62630`.
Evidence: `../results/shardstream-refactor-20261009/native-consumer/qualification.json`.

The TP4→TP2×8 case completes 40 steps, passes all 1,056 steady native cache
records and is locally archived with verified SHA-256. Historical native
profiles for this older comparison record four GIN contexts, while the current
`6dabfbf` baseline and ShardStream both use eight. Its latency improvement must
not be attributed to this refactor. The adjacent Elastic comparison establishes
preservation of the current baseline on 22 resource metrics; historical
workflow comparisons supply the broader performance checks requested by the user.

## User-directed topology exclusion

On 2026-10-09 the user explicitly excluded PP4 as an unsuitable topology.
The active PP4→TP2 case was cancelled, and PP4→TP2/TP4/TP8 were removed
from the launch queue and required acceptance coverage. Their old controls
remain historical records; the cancelled case is not a new-library pass.
The two PP1 FP8 configurations subsequently completed. No additional
Elastic rerun is queued. All 21 standalone cases pass correctness, cache and generation checks; final-package FP8 naive performance remains unresolved.

## Real FP8 GRPO qualification at `3dd7325`

| Training TP → inference TP2×8 | Full update p50 / p95 | Historical transport update | Trainer step p50 / p95 | Historical step |
|---|---:|---:|---:|---:|
| TP2/PP1/CP2/EP8 FSDP | 314.19 / 323.15 ms | 341.77 / 347.90 ms | 21.66 / 23.55 s | 21.77 / 23.05 s |
| TP4/PP1/CP2/EP8 FSDP | 309.85 / 315.17 ms | 311.10 / 319.59 ms | 25.34 / 28.95 s | 24.64 / 27.92 s |

Both cases complete 10 actual GRPO steps, eight steady publications and seven
complete cycles, exit 0. All 256 steady native records per case cover exactly
32 ranks per version, use FP8, hit cache and report zero the five preparation/
rebuild costs. Ten transfer resource configuration values retain the original
settings. Full raw archives are locally SHA-256 verified. Historical controls
are explicitly reused; these are workflow comparisons, not causal attribution
to the refactor. The largest step change is +3.70% p95 in the TP4 case.

Evidence: `../results/shardstream-refactor-20261009/historical-fp8-grpo-comparison.json`
and both `grpo-fp8-matrix1/*/native-audit.json` records.

## User-defined naive routing revision

The user clarified naive as a shared fixed entry instance followed by relays
within each node, then the next node. Previously fixed engine-ID order alternated
nodes under round-robin placement. Naive now groups actual placement without
source-dependent rotation. Ordinary writer/reader setup collects actual node IDs
before preparing its host plan, including partially occupied rollout nodes.
Swizzle also consumes actual placement; off stays direct. All native sources and
resource configuration defaults remain unchanged. The earlier naive results
measure the previous route definition and are retained as historical evidence.

A new complete 21-case standalone BF16/FP8 × off/naive/swizzle matrix is pending.
Each worker records actual ring edges for path audit, in addition to full loaded
HF weights, exact FP8 byte/scale checks, cache and generation validation. PP4 and
additional Elastic runs remain excluded. CPU regressions: 79 pass.
