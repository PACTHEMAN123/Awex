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
