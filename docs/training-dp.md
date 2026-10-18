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
