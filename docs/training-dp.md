# Replicated training source selection

`ParameterMeta.replicas` already describes complete logical copies of each
parameter. A copy may consist of several tensor-parallel shards. The original
planner selects one training copy for each parameter and inference replica.

The optional block policy instead uses one replica to establish logical shard
geometry, finds equivalent physical shards across training replicas, and assigns
disjoint destination blocks to those sources. It balances cumulative source bytes
across parameters. Each target element still has exactly one source. Sources
must belong to the same completed training snapshot, as for the existing replica
policy.

The implementation retains the existing `CommunicationOperation` representation
and transport kernels. Block size controls source assignment and is independent
of transport chunk size. Rows remain intact for the existing contiguous and
strided-copy lowering. Parameters with different shard geometries across replicas
retain the replica policy. Training DP1 also retains the replica policy.

With ring broadcast, the same logical block uses the same source across all
rollout engines. Each source therefore has identical streams for corresponding
rollout workers, preserving a single injection followed by ring forwarding.
Source assignment happens before local-rank pruning, so senders and receivers
independently produce matching plans.

## Configuration

The default policy remains `replica` while the new policy is evaluated. Set:

```bash
export SHARDSTREAM_DP_SOURCE_POLICY=block
export SHARDSTREAM_DP_SOURCE_BLOCK_BYTES=1048576
```

The standalone recipe runner exposes the same settings as
`--dp-source-policy block --dp-source-block-mb 1`. The environment default block
size is 16 MiB. Invalid policies and nonpositive block sizes are rejected.

`SHARDSTREAM_RING_SOURCE_NODE_SWIZZLE=1` optionally spreads ring entry nodes
according to actual training-node placement. Corresponding DP ranks can differ
by an even node size, causing the original rank-based swizzle to choose the same
rollout entry node. The new strategy uses source-node order for the entry node
and node-local source order for colocated engines. All ranks gather this placement
once per membership, and sender and receiver routes use the same seed. Enable it
in the standalone runner with `--dp-source-ring-order source-node`; `rank` remains
the default. This option uses the existing ring swizzle mode and kernels.

## Partitioning corresponding shard streams

`--dp-source-policy stream` groups parameter shards by their equivalent source
ranks. Each group traverses parameters in name/geometry order and partitions
the cumulative canonical-shard bytes equally among DP replicas. Within a DP
range, other equivalent copies (including CP copies) divide the range equally.
Only a partition boundary inside a parameter creates an extra row-aligned slice;
a boundary on a parameter edge leaves both parameters intact.
No contiguous staging buffer is required,
and the block-size setting does not control stream partitioning.

Partitioning happens independently for every corresponding TP/EP source group,
before pruning a local plan. DP replicas may each span multiple physical nodes;
the algorithm uses rank membership, rather than treating a DP replica as a node.
Different replica geometries, DP1, and incomplete DP identities retain the
existing replica planner. `SHARDSTREAM_DP_STREAM_PLAN` reports canonical source
bytes, group counts and boundary splits. Actual injected bytes can additionally
include replicated inference parameter views and are reported by transport.

`--dp-source-ring-order dp` (environment `SHARDSTREAM_RING_DP_ENTRY=1`) binds
DP replica i to rollout node i in first-engine order. It uses each operation's
source DP identity on both endpoints, so all nodes of one DP replica select the
same rollout entry node. Colocated rollout engines can still rotate within that
node. This explicit pairing requires ring broadcast/swizzle and at least one
rollout node per active DP identity; the standalone recipes enable both ring
features. Source identity conflicts are rejected. The default policy and entry
order remain `replica` and `rank`.

## Qwen3-30B-A3B experiments

The existing layouts `qwen30-tp4-cp2-ep8-r2tp8ep8` and
`qwen30-tp4-cp2-ep8-r4tp4` already have training DP2: both use 16 training GPUs,
TP4, PP1, CP2, EP8, and expert TP1. Dense DP is `16/(4*1*2)=2`; expert DP is
`16/(1*1*8)=2`. Each 8-GPU training node contains TP4 × CP2 and all eight EP
ranks. Rollout occupies two different 8-GPU nodes, with one TP8/EP8 engine or two
TP4 engines on each node.

The October 18 supplement compares the two policies on the same source/native
revision and preserves earlier experiment results. Every launch checks actual
training rank placement, every rollout parameter and publication version,
parameter storage stability, deterministic generation before/after publication,
and counters on all four selected HCAs of each participating node. Formal runs
use five warmup publications and thirty measured publications, with three
independent launches per policy and topology. Screening runs are reported
separately.

Experiment artifacts are collected under
`results/training-dp-20261018` in the workspace and
`/mnt/fuse/verl-e2e/runs/training-dp-20261018` on the experiment head.

## Measured results (2026-10-18)

These are real BF16 Qwen3-30B-A3B publications on four eight-GPU H20-3e nodes.
Each configuration has three independent launches, five warmup publications
and thirty measured publications per launch. Values below are the median of
the three launch p50s and the median of their p95s, in milliseconds.
The matched replica controls use the original rank-based ring order.

| Rollout | Replica p50 / p95 | Candidate p50 / p95 | p50 change | Candidate |
|---|---:|---:|---:|---|
| 2 × TP8/EP8, one per node | 606.90 / 650.03 | 429.81 / 451.36 | -29.2% | 1 MiB blocks, source-node entry |
| 4 × TP4, two per node | 389.81 / 395.37 | 374.76 / 388.49 | -3.9% | 1 MiB blocks, rank entry |

The [launch data](data/qwen30-training-dp2-20261018-launches.csv) includes
every independent result, cold publication latency, dense/expert DP sizes,
source revision and screening configurations, including regressions.
The [formal aggregates](data/qwen30-training-dp2-20261018-formal.json) retain
each launch p50 and p95. Screening uses only seven measured publications and
is separate from the formal comparison. The TP8 combined candidate changes
both source granularity and ring entry; its complete benefit must not be
attributed to block load balancing alone.
A single-launch TP8 route-only screening measured replica/source-node at
451.10 ms and block/source-node at 449.30 ms. This points to entry-node
distribution as the main TP8 gain; extra benefit from block selection alone
is not established. The small TP4 change overlaps launch-to-launch variation.

All 510 successful publications passed full loaded-parameter equality after
clearing destinations, version-marker checks and parameter-address stability.
Every rollout engine passed deterministic generation before/after publication.
Every measured rank reused its device plan with no batch rebuilding, metadata
upload or transport initialization. Actual placement confirms one member of
each dense/expert DP2 group per training node; all sixteen training ranks send.
All four HCAs on each participating node carried traffic.

One additional TP4 attempt failed before measurement when accumulated NCCL
temporary segments filled `/dev/shm`. Its four-node logs were retained; only
closed NCCL segments created during this task were reclaimed. That attempt
was retried with identical code and transport settings and contributes no
performance samples. Previous successful launches were reused.

Formal benchmark revision: `aa5869911044d80a1102d23f344bb163a23740ca`.
The final code validation at `48adb09` additionally covers partial blocks,
geometry fallback, the DP2 recipes and source-node routing after membership
changes; each node passed 93 tests (9.86–10.31 seconds).
The unchanged native extension SHA-256 is
`d134ab2c2ac9471fb2c55c563565e55dd9954d95e6ef7c847c422e041ec49c6c`.

Full raw artifacts, per-publication CSV, manifests, route-only screening and
the historical r23 comparison are in the workspace supplement
`results/training-dp-20261018/REPORT.md`. The raw bundle was downloaded and
verified against the head SHA-256:
`4a1c7237bef75fbcbe9b303b91b85ede62199298fca82d94fc1e62d08625311f`.
OSS FUSE writes returned zero-byte files and EIO on readback; the complete local evidence bundle is the verified persistent copy.

## Stream partition measurements (2026-10-18)

The selected configuration uses cumulative Stream partitioning and explicit
DP-to-rollout-node entry pairing. Each topology has three independent
launches, five warmup publications and thirty measured publications per
launch. Values are medians of launch p50/p95 values, in milliseconds.
This section records Stream performance only. Historical experiments above
remain unchanged.

| Rollout | Stream p50 | Stream p95 |
|---|---:|---:|
| 2 × TP8/EP8 | 418.04 | 449.93 |
| 4 × TP4 | 419.32 | 448.12 |

| Rollout | Three launch p50 values | Three launch p95 values |
|---|---|---|
| 2 × TP8/EP8 | 418.04, 413.36, 435.43 | 450.80, 429.95, 449.93 |
| 4 × TP4 | 439.02, 363.72, 419.32 | 448.12, 386.57, 458.56 |

TP4 has substantial launch variation; all three points are retained.

| Rollout | Groups / source shard regions / boundary splits | Writer spans | Max/min writer bytes |
|---|---:|---:|---:|
| 2 × TP8/EP8 | 13 / 19449 / 27 | 22044 | 1.005616 |
| 4 × TP4 | 13 / 19449 / 27 | 75540 | 1.000000 |

Canonical source regions are counted once; replicated inference parameter
views can add injections. Writer bytes/spans above are observed transport
values, so storage-layout fragments and destination resharding are included.
All 210 Stream publications passed full loaded-weight equality, version
and address checks; deterministic generation agreed before/after publication.
The audit also verified every DP entry node and complete forwarding chain,
steady plan reuse, and four-HCA traffic on every node. Each node passed
107 CPU tests.

Runtime/test revision: `e8bd224`. Native binary remains unchanged. The actual
GPU layout has one training DP per node; CPU tests additionally exercise a DP
spanning multiple nodes. Unused NCCL shared-memory segments were reclaimed
under the global benchmark lock with user authorization; each launch then
reclaimed only its own new closed segments. All 32 GPUs were released.

Use `--dp-source-policy stream --dp-source-ring-order dp`.

Independent points: [Stream launch CSV](data/qwen30-training-dp2-stream-20261018-launches.csv)
and [Stream aggregates](data/qwen30-training-dp2-stream-20261018-formal.json).
Raw evidence and reproduction scripts are in the workspace supplement
`results/training-dp-stream-20261018/REPORT.md`.
Verified evidence SHA-256: `1a1fa3393b0155317f82cb4394e0194620ed5b1b271a71a56ddd6b3039675a23`.

## Additional TP4 Stream launches (2026-10-18)

Five additional independent launches use exactly the Stream + DP entry
configuration of the original 363.72 ms p50 / 386.57 ms p95 run. Each launch
has five warmup and thirty measured publications. No algorithm or transport
configuration changed; runtime remains `e8bd224`. All points are retained.

| Additional launch | p50 (ms) | p95 (ms) |
|---|---:|---:|
| 1 | 446.39 | 492.62 |
| 2 | 410.16 | 412.93 |
| 3 | 413.95 | 434.02 |
| 4 | 368.35 | 430.98 |
| 5 | 372.54 | 402.17 |

The five-launch medians are 410.16 ms p50 and 430.98 ms p95.
Including the original three launches, eight-launch medians are 412.05 ms
p50 and 432.50 ms p95. New runs verified 175 full publications
(150 measured); all eight verified 280 publications (240 measured). Routes,
plan reuse and four-HCA traffic passed audit. All 32 GPUs were released.

[All eight launches](data/qwen30-training-dp2-stream-tp4-repeat-20261018-launches.csv)
and [batch summaries](data/qwen30-training-dp2-stream-tp4-repeat-20261018-summary.json).
Raw supplement: `results/training-dp-stream-tp4-repeat-20261018/REPORT.md`.

### Interpretation

The historical per-tensor/rank-entry TP4 median was 389.81 ms; the current
Stream/DP-entry eight-launch median is 412.05 ms, about 5.7% higher
latency. TP8 improved from 606.90 ms to 418.04 ms across the combined changes,
but this comparison changes both source partition and ring entry. An earlier
single-launch TP8 screening kept per-tensor selection and changed rank entry
to source-node entry: 535.38 ms to 451.10 ms. This supports entry spreading,
with the limits of a single screening and a different entry implementation.
There is no formal repeated per-tensor control using the same DP entry, so
the evidence does not establish a standalone Stream partition advantage.
No additional control experiments were run in this supplement.

### Previously measured Qwen30B training DP

Dense DP = world / (TP × PP × CP); expert DP = world / (ETP × PP × EP).
All configurations below use expert TP1. These are training replica sizes.

| Training GPUs | TP / PP / CP / EP | Rollout | Dense DP | Expert DP |
|---:|---|---|---:|---:|
| 16 | 2 / 1 / 2 / 8 | 2 × TP4/EP1 | 4 | 2 |
| 16 | 2 / 1 / 2 / 8 | 4 × TP4/EP1 | 4 | 2 |
| 8 | 2 / 1 / 1 / 8 | 6 × TP4/EP1 | 4 | 1 |
| 16 | 2 / 4 / 2 / 4 | 4 × TP4/EP1 | 1 | 1 |
| 16 | 4 / 1 / 2 / 8 | 2 × TP8/EP8 | 2 | 2 |
| 16 | 4 / 1 / 2 / 8 | 4 × TP4/EP1 | 2 | 2 |

The 16-training-GPU / 2×TP4 rollout row is an Awex-only historical point.
The fully-async GRPO rollout experiments separately use fixed 16-GPU
TP2/PP1/CP2/EP8 training, hence dense DP4 and expert DP2 for all four rollout
layouts. Stream GPU validation currently covers the TP4/CP2/EP8 DP2 rows.
[Successful outcome sources](data/qwen30-training-dp-topologies.json).
