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

Within one LSA domain, a sender's worker warps stage source data into its local
registered FIFO window and publish `ready_step`; receiver warps issue volatile
loads from that remote window, scatter into local tensor fragments, then
publish `consumed_step` back to the sender.

Across nodes, NCCL commonly exposes network connections only between matching
local ranks (`NCCL_GIN_CONNECTION_RAIL`). V2 preserves arbitrary logical peer
transfers with a two-hop route: the sender writes over its GIN rail into a
proxy window on the destination node, then the receiver reads that proxy over
LSA/NVLink. Acknowledgements take the reverse GIN-rail-plus-LSA route. A hybrid
world barrier resets the ordinary-memory FIFO tokens safely between recurrent
publications, and `NCCL_WIN_STRICT_ORDERING` orders each payload write before
its ready token. This avoids requiring a full-mesh GIN topology or changing
the Python `TransferPlan`.

Every `(peer, channel)` owns an independent step stream. Eight 512 KiB FIFO
steps provide 4 MiB of in-flight payload per channel-peer connection. The H20
four-node recipe caps execution at eight channels, which keeps the 32-rank
dense proxy window near 1 GiB per GPU.

The Python transport entry point lives next to this directory in
`awex/transfer/nccl_device_v2.py`. It has its own task binding and extension
loader; the existing v1 transport is not used as a compatibility layer. The
top-level writer/reader route to this backend only when
`comm_backend=nccl_device_v2`.
