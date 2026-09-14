# NCCL Device v2 draft

This directory is an isolated device-side draft. It does not change the
existing `TransferPlan` or the v1 extension.

The input task order remains the source of truth. The backend only lowers each
fixed task into smaller `V2Chunk`-like `V2Work` records. The CUDA execution
hierarchy follows the useful NCCL P2P shape:

```text
fixed task
  -> chunk
  -> channel work batch
  -> NCCL P2P size formula selects active channels
  -> one 640-thread CUDA CTA per channel
  -> 20 warps divided across the active works in a batch
  -> 16-byte vector copy, unrolled across 8 instructions
  -> FIFO slot = absolute step % 8
```

For intra-node SIMPLE traffic, work channel count follows NCCL's
`addP2pToPlan` sizing: `minPartSize = stepSize / 8`,
`maxPartSize = stepSize * 32`, bounded by the power-of-two channel capacity of
the GPU. Work groups use CUDA named barriers, leaving barrier 0 to CTA-wide
synchronization. A full eight-work batch gives each work two warps and maps the
eight independent work steps onto all eight FIFO generations.

The FIFO payload is a registered window slot. The control protocol is separate
from task ordering: every `(peer, channel)` owns an independent monotonically
increasing step stream, and each launch receives the next absolute step from
the host lowering state.

The Python transport entry point lives next to this directory in
`awex/transfer/nccl_device_v2.py`. It has its own task binding and extension
loader; the existing v1 transport is not used as a compatibility layer. The
top-level writer/reader route to this backend only when
`comm_backend=nccl_device_v2`.
