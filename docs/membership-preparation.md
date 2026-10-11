# Measuring and reducing membership preparation

Dynamic model joins prepare every old and new participant before the next
publication. Model loading happens before this interval. Preparation includes
the metadata process group, actual rollout topology, old transport teardown,
the independent native NCCL communicator, FIFO windows, GIN resources, routes,
and device plan uploads. A ready barrier completes before acknowledgement.

`prepare_model_membership` returns per-worker host intervals in `phases`.
`preparation_metrics.native_preparation_phases_ms` records native communicator,
topology, feature/geometry exchanges, allocation, registration, GIN creation,
and channel negotiation. `transport_initialization_phases_ms` additionally
records extension loading, feature validation, identity exchange, unique ID
exchange, and native creation. These scopes overlap: native creation is nested
inside transport initialization, and plan initialization contains window/GIN
preparation. Do not add nested durations or different workers' maxima to claim
a critical path. Use coordinator wall time and disjoint intervals on one worker.

The same loaded model storage survives a join. Preparation never launches a
weight transfer kernel or advances a weight version. The first publication
after a join must hit the already prepared plan cache.

## Isolated runtime comparison

The validated default for subsequent CUDA 13 H20 runs is NCCL 2.31.2.
`requirements/nccl-cu13.txt` pins the package and `scripts/nccl-default.sh`
checks both the headers and library version before exporting the build/runtime
paths. Build the native extension against that installation's Device API
headers and load the matching library before PyTorch. For an explicit comparison
installation:

```sh
export SHARDSTREAM_NCCL_LIB=/path/to/nccl/lib
export LD_PRELOAD="$SHARDSTREAM_NCCL_LIB/libnccl.so.2"
export LD_LIBRARY_PATH="$SHARDSTREAM_NCCL_LIB:$LD_LIBRARY_PATH"
```

NCCL 2.30.7 provides explicit strong/weak signal semantics and optimizes GDAKI
QP overhead when counters are unused. The transport already selects the
corresponding signal APIs and requests strong signals when compiled against
2.30.7 or newer. Validate complete parameter contents, first-publication cache
hits, and steady publication latency when changing the runtime.

NCCL 2.31.2 also batches GDAKI QP allocation. On the four-node H20,
Qwen3-30B-A3B experiment, the current 2.30.4 runtime prepared the 2-to-4
and 4-to-8 engine joins in 18.173 / 21.973 seconds. A matched 2.31.2
header/runtime build with the default control group measured 10.003 /
11.758 seconds. Native connections, contexts, FIFO/window geometry, channels
and signal counts matched across every rank. This is a measured runtime
comparison, not a change to the transport resource configuration.

Each setting had one complete elastic run; sequential runs can drift. Full
parameter contents and preparation cache hits passed. The upgrade kept
background GEMM work active during preparation; actual optimizer/serving
performance needs application-specific validation.

See [NCCL 2.30.7 release](https://github.com/NVIDIA/nccl/releases/tag/v2.30.7-1)
and [2.31.2 GDAKI implementation](https://raw.githubusercontent.com/NVIDIA/nccl/v2.31.2-1/src/transport/net_ib/gdaki/gin_host_gdaki.cc)
for the upstream changes.

## Optional control group budget

For dynamic model membership, `SHARDSTREAM_CONTROL_GROUP_CTAS=1` passes
`min_ctas=max_ctas=1` through PyTorch's NCCL process-group options. This group
only exchanges metadata and executes publication/readiness barriers. It is
independent of the native weight-transfer communicator; native channels,
GIN connections/contexts, FIFO depths and steps remain configured separately.
The option is unset by default. It did not improve both joins consistently
in the runtime comparison, so keep the default when adopting the upgrade. It requires a PyTorch NCCL configuration API
that exposes those fields. Unsupported settings fail explicitly.

The experiment controller, rank-level measurements, raw logs and checksum
audits for the 2026-10-11 four-node H20 comparison are saved outside this
repository in `results/shardstream-prepare-20261011` in the parent workspace.
