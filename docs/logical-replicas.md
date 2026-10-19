# Logical replica source selection and forwarding

Enable the opt-in planner on every training and inference rank:

```bash
export SHARDSTREAM_REPLICA_ROUTING=logical
export SHARDSTREAM_DP_SOURCE_POLICY=stream
export SHARDSTREAM_RING_BROADCAST=1
export SHARDSTREAM_RING_SWIZZLE=1
```

The standalone recipe runner exposes `--replica-routing logical` and
`--dp-source-policy stream`. The legacy `engine` routing remains the default.
Logical routing also supports the `replica` source policy; the historical
fixed-size `block` policy is rejected with this mode.

## Ownership and source partitioning

For each canonical parameter, collect physical source and destination owners.
Their logical offset and shape boundaries define disjoint rectangular regions.
Every region records all training ranks containing it and every inference rank
requiring it. This includes training DP, CP and replicated model parameters,
inference DP, TP-replicated parameter views, and independent rollout engines.
Source replicas may have different shard geometries. Missing training coverage,
conflicting physical ownership and incompatible global shapes are rejected.

Regions with the same candidate ranks and DP/CP identities form a deterministic
cumulative byte stream across parameters. Stream partitions divide the bytes
equally between complete DP groups, then between equivalent owners within a DP
group. Where the DP identities do not describe a complete group (for example
expert ownership or model-internal copies), physical owners divide the stream.
Only partition boundaries inside a region create extra row-aligned slices.
Every slice binds its own parameter metadata and native source storage; there
is no packed model buffer. All source copies must belong to the same completed
training snapshot. FP8 ownership and source cuts must preserve whole tiles.

## Physical routes

Each selected source interval has an explicit destination set, independent of
engine boundaries. Intervals sharing a source and destination set reuse a route
ID, allowing native lowering to concatenate parameter spans into one stream.
The small global route catalog survives local-plan pruning. Each participant
therefore makes the same routing decision from the physical rank-to-node map.

During preparation, compare source-placement routing with a greedy schedule
that visits routes from largest to smallest and chooses the eligible node with
the fewest assigned entry bytes. Source placement breaks ties. Keep the existing
entry mapping unless the greedy schedule reduces the difference between the
largest and smallest node loads. All receivers on a node are consecutive before the chain crosses
to another node. The source retains only its head operation; every receiver
records its predecessor and successor and binds its own destination tensor.
With uniform supported destination types, each logical interval is injected
once and enters each required node once. Local copies use the existing NCCL
LSA/NVLink path where the ranks share an LSA team. Entry balancing is greedy
at route granularity; indivisible routes and unequal destination sets can
prevent equal entry loads.

Physical placement is gathered once per membership, including individual ranks
of engines spanning nodes. Prepared batches and native device plans are reused
for subsequent publications. There is no publication-time scheduling collective.
The existing CUDA kernels and native binary are unchanged.

The catalog ID is independent of the native ring ID used to select channels.
Routes with one corresponding worker in each engine keep the established
source/TP ring encoding. Expanded destination sets use a disjoint ID range.
This preserves channel placement for unchanged streams while allowing new
within-engine replica chains.

`SHARDSTREAM_LOGICAL_REPLICA_PLAN` reports unique source bytes, total destination
bytes, source groups, ownership regions and boundary splits.
`SHARDSTREAM_LOGICAL_REPLICA_ENTRIES` reports assigned entry bytes by physical
node. Existing route, span, cache and per-publication profiling records remain
available. A logical plan hash includes both explicit destination sets and the
global route catalog.

## Validation

`tests/test_logical_replica_routes.py` checks consecutive updates through the
actual prepared Python descriptors, exact destination coverage, local/global
plan agreement, source partitions, entry loads and host-consecutive chains.
It covers training DP1/2/4, CP, model-internal copies without complete DP
identities, inference DP, multiple engines and differing replica geometries.
CPU checks establish plan/descriptor correctness; GPU publication measurements
are reported separately with their exact source revision and topology.

The standalone vLLM entrypoint explicitly loads a worker extension so fresh DP
processes register the same ShardStream RPCs as forked TP workers. The recipe
runner's `--disable-prefix-caching` option makes before/repeat/after generation
checks use the same uncached prefill shape. It still checks exact outputs and
pins requests to DP rank zero. Model-weight profiling additionally clears and
exactly compares every parameter on every inference worker outside the timed
publication, checks version markers, and verifies stable parameter addresses.
