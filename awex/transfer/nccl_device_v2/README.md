# NCCL Device v2 draft

This directory is an isolated device-side draft. It does not change the
existing `TransferPlan` or the v1 extension.

The input peer, direction, and task order remain the source of truth. For each
peer, C++ lowering concatenates the ordered physical tensor spans into one
virtual, discontinuous byte stream. It partitions that whole stream across
channels before creating `V2Work` chunks, so a chunk may contain fragments
from multiple tensors. Python does not perform transport chunking.

The CUDA execution hierarchy follows the useful NCCL P2P shape:

```text
fixed TransferPlan spans for one peer
  -> one virtual discontinuous stream
  -> topology/bandwidth channel selection and 4 KiB-aligned channel parts
  -> C++ 4 MiB work chunks, each containing one or more tensor fragments
  -> channel work batch
  -> one 640-thread CUDA CTA per channel
  -> 20 warps divided across the active works in a batch
  -> explicit WaitSend/WaitRecv, worker, and PostSend/PostRecv roles
  -> volatile 16-byte vector loads, unrolled across 8 instructions
  -> FIFO slot = absolute step % 8
```

For intra-node SIMPLE traffic, work channel count follows NCCL's
`addP2pToPlan` sizing: `minPartSize = stepSize / 8`,
`maxPartSize = stepSize * 32`. The topology layer mirrors NCCL's NVLink path
formula, `2 * max(1, pathBandwidth / linkBandwidth)`. It rounds the per-peer
path demand up to a power of two, then caps it by the configured channel
ceiling and the device's SM capacity. The raw, requested, and effective values
are exposed in launch metrics. NCCL's internal collective-graph channel count
is not used as a v2 execution cap because this backend has a different CTA
shape. Work groups use CUDA named barriers,
leaving barrier 0 to CTA-wide synchronization. When a work has at least three
warps, its final warp is reserved for the Post role so it can publish the
previous step while worker warps begin the next one.

The backend selects a transport per peer without changing the fixed plan or
the public `nccl_device_v2` backend name. Peers in the local LSA team keep the
NVLink read protocol: a sender stages into its local FIFO, the receiver reads
that window, and `ready_step`/`consumed_step` carry ownership. Peers outside the
LSA team use GIN puts into the receiver's FIFO. Strong ready signals order the
put stream and weak credit signals release reusable sender slots. Every
`(peer, channel)` still owns an independent monotonically increasing step
stream, and both paths share the same work/channel lowering and symmetric
registered window.

GIN is initialized only when the cached plan contains a non-LSA edge. That
path requires Linux, CUDA 12.2 or newer, NCCL 2.30.4 or newer with aggregate
`nccl_device.h` headers, a GIN-capable communicator, and a fully connected
supported RDMA fabric. The launch metrics expose `lsa_peer_count`,
`gin_peer_count`, `gin_type`, and `gin_context_count` so a deployment can
confirm which path was selected. Indexed GIN signals are reset behind a world
barrier before each cached-plan launch.

The Python transport entry point lives next to this directory in
`awex/transfer/nccl_device_v2.py`. It has its own task binding and extension
loader; the existing v1 transport is not used as a compatibility layer. The
top-level writer/reader route to this backend only when
`comm_backend=nccl_device_v2`.
