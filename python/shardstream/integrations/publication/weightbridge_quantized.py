"""Experimental gather-before-quantize frontend for native WeightBridge.

This is NOT upstream WeightBridge support or a framework integration. Uniform,
block-aligned 2-D TP partitions are supported. Every publication gathers each
physical tensor on every trainer, quantizes the complete tensor, then exposes
the original partition as FP8 bytes plus FP32 scales to native specgen/RDMA.
All tensors are prepared before send_weights(), not pipelined with RDMA rounds.
"""

from dataclasses import dataclass
import time

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class MatrixPartition:
    shape: tuple[int, int]
    axis: int
    rank: int
    world_size: int
    block: int = 128

    def __post_init__(self):
        if (len(self.shape) != 2 or min(self.shape) <= 0 or self.axis not in (0, 1)
                or self.world_size <= 0 or not 0 <= self.rank < self.world_size
                or self.block <= 0):
            raise ValueError("Invalid matrix partition")
        if any(n % self.block for n in self.shape):
            raise ValueError("Full matrix dimensions must be block aligned")
        if self.shape[self.axis] % (self.world_size * self.block):
            raise ValueError("TP shard boundaries must be block aligned and uniform")

    def slices(self, *, scales=False):
        divisor = self.block if scales else 1
        width = self.shape[self.axis] // self.world_size // divisor
        result = [slice(None), slice(None)]
        result[self.axis] = slice(self.rank * width, (self.rank + 1) * width)
        return tuple(result)

    @property
    def local_shape(self):
        result = list(self.shape)
        result[self.axis] //= self.world_size
        return tuple(result)

    @property
    def logical_gather_egress_bytes(self):
        # Ring-allgather useful payload per rank, excluding protocol headers.
        return 2 * self.shape[0] * self.shape[1] * (self.world_size - 1) // self.world_size


class GatherBeforeQuantize:
    """Own persistent staging; consume live BF16 shards on every prepare().

    All ranks must provide the same names/shapes/axes in the same process group.
    The quantizer must return E4M3FN values and FP32 dequantization scales.
    Exposing FP8 storage as uint8 avoids numeric conversion by native copies
    and allows the unchanged bit-pattern correspondence extractor to probe it.
    """

    def __init__(self, shards, partitions, quantizer, group=None):
        if not shards or set(shards) != set(partitions):
            raise ValueError("Shards and partitions must have identical nonempty keys")
        self.shards = shards
        self.partitions = partitions
        self.quantizer = quantizer
        self.group = group
        self.payload = {}
        self.gathered = {}
        self.full = {}
        self._pointers = {}
        devices = {t.device for t in shards.values()}
        if len(devices) != 1:
            raise ValueError("All shards must share one device")
        self.device = next(iter(devices))
        for name in sorted(shards):
            x, p = shards[name], partitions[name]
            if x.dtype != torch.bfloat16 or tuple(x.shape) != p.local_shape:
                raise ValueError(f"Invalid BF16 shard: {name}")
            if not x.is_contiguous():
                raise ValueError(f"Shard must be contiguous: {name}")
            if p.world_size != dist.get_world_size(group) or p.rank != dist.get_rank(group):
                raise ValueError("Partition rank/world size differs from gather group")
            self._pointers[name] = x.data_ptr()
            self.gathered[name] = torch.empty(x.numel() * p.world_size, dtype=x.dtype, device=x.device)
            self.full[name] = (self.gathered[name].view(p.shape) if p.axis == 0 else
                               torch.empty(p.shape, dtype=x.dtype, device=x.device))
            self.payload[name] = torch.empty(p.local_shape, dtype=torch.uint8, device=x.device)
            self.payload[name + ".scale"] = torch.empty(
                tuple(n // p.block for n in p.local_shape), dtype=torch.float32, device=x.device)
        metadata = [(name, p.shape, p.axis, p.world_size, p.block)
                    for name, p in sorted(partitions.items())]
        peers = [None] * dist.get_world_size(group)
        dist.all_gather_object(peers, metadata, group=group)
        if any(peer != metadata for peer in peers):
            raise ValueError("All trainers must agree on tensor metadata/order")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def prepare(self):
        """Blocking preparation; caller includes it in publication latency."""
        self._sync()
        begin = time.perf_counter()
        gather_s = quantize_s = 0.0
        for name in sorted(self.shards):
            x, p = self.shards[name], self.partitions[name]
            if x.data_ptr() != self._pointers[name]:
                raise RuntimeError("Live source storage changed; rebuild the frontend")
            start = time.perf_counter()
            dist.all_gather_into_tensor(self.gathered[name], x.view(-1), group=self.group)
            full = self.full[name]
            if p.axis == 1:
                torch.cat(self.gathered[name].view(p.world_size, *p.local_shape).unbind(0),
                          dim=1, out=full)
            self._sync()
            gather_s += time.perf_counter() - start
            start = time.perf_counter()
            q, s = self.quantizer(full, (p.block, p.block))
            if (q.dtype != torch.float8_e4m3fn or tuple(q.shape) != p.shape
                    or s.dtype != torch.float32
                    or tuple(s.shape) != tuple(n // p.block for n in p.shape)):
                raise ValueError("Quantizer returned an incompatible payload")
            self.payload[name].copy_(q.view(torch.uint8)[p.slices()])
            self.payload[name + ".scale"].copy_(s[p.slices(scales=True)])
            self._sync()
            quantize_s += time.perf_counter() - start
            del q, s
        return dict(prepare_ms=(time.perf_counter() - begin) * 1000,
                    gather_ms=gather_s * 1000, quantize_ms=quantize_s * 1000,
                    logical_gather_egress_bytes=sum(
                        p.logical_gather_egress_bytes for p in self.partitions.values()))

    def publish(self, adapter):
        """Prepare, native RDMA send, and wait for complete receiver consumption."""
        self._sync()
        begin = time.perf_counter()
        stats = self.prepare()
        start = time.perf_counter()
        event = adapter.send_weights()
        if event is not None:
            event.synchronize()
        adapter.wait_send_complete()
        stats.update(native_transfer_ms=(time.perf_counter() - start) * 1000,
                     complete_ms=(time.perf_counter() - begin) * 1000)
        return stats
