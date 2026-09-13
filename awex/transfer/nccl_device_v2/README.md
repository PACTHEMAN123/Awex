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
  -> one CUDA CTA per channel
  -> one warp per active send/recv side
  -> FIFO step = absolute step % fifo_depth
```

The draft intentionally starts with one warp per active side. This keeps the
work-level synchronization local to a warp while the channel, batch, and step
protocol are being validated. A later pass can add NCCL's multi-warp
`nWarpPerWork` path using named barriers without changing the host task model.

The FIFO payload is a registered window slot. The control protocol is separate
from task ordering: every `(peer, channel)` owns an independent monotonically
increasing step stream, and each launch receives the next absolute step from
the host lowering state.

The Python transport entry point lives next to this directory in
`awex/transfer/nccl_device_v2.py`. It has its own task binding and extension
loader; the existing v1 transport is not used as a compatibility layer. The
top-level writer/reader route to this backend only when
`comm_backend=nccl_device_v2`.
