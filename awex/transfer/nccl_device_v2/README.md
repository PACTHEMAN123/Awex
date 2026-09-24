# NCCL Device v2 draft

This directory is an isolated device-side draft. It does not change the
existing `TransferPlan` or the v1 extension.

The input peer, direction, and task order remain the source of truth. For each
peer, C++ lowering concatenates the ordered physical tensor spans into one
virtual, discontinuous byte stream. It partitions that whole stream across
channels before creating `V2Work` chunks, so a chunk may contain fragments
from multiple tensors. Python does not perform transport chunking.

The stream uses the receiver tensor dtype as its wire format. When a sender
tensor has a different supported floating-point dtype, worker warps cast each
element while filling the current FIFO step. No full-size converted tensor is
materialized. The receiver writes the encoded bytes directly into its target
tensor, so BF16-to-FP8 uses half the payload bytes. FP16, BF16, FP32, FP8 E4M3,
and FP8 E5M2 participate in streaming casts; equal-dtype and other opaque
equal-size transfers retain the vectorized byte-copy path. Scale computation
is deliberately outside this primitive: scaled quantization formats expose
their scale tensors as ordinary plan entries and select the desired wire dtype
for the associated weight tensor.

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
  -> volatile 16-byte vector copy or per-element streaming cast
  -> FIFO slot = absolute step % 8
```

The Hopper block-wise FP8 path uses a separate 1024-thread CTA. Fifteen and
sixteen worker warps quantize two 128 x 128 shared-memory stages concurrently;
the remaining warp owns FIFO control, TMA refill, and ready-step publication.
Only group-level named barriers participate in the tile loop.

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

The transport is fixed to NVLink read mode. A sender's worker warps stage source
data into its local registered FIFO window and publish `ready_step`; receiver
warps issue volatile loads from that remote window, scatter into local tensor
fragments, then publish `consumed_step` back to the sender. Every
`(peer, channel)` owns an independent monotonically increasing step stream.
Wait roles cache observed steps and poll with volatile loads. Post roles use a
system fence followed by a relaxed system store, avoiding system-scope RMWs on
the normal control path. Eight 512 KiB FIFO steps provide 4 MiB of in-flight
payload per channel-peer connection.

The Python transport entry point lives next to this directory in
`awex/transfer/nccl_device_v2.py`. It has its own task binding and extension
loader; the existing v1 transport is not used as a compatibility layer. The
top-level writer/reader route to this backend only when
`comm_backend=nccl_device_v2`.
