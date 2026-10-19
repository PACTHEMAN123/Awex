"""Partition logical ownership regions and describe their complete replica chains.

The grid is defined by ownership boundaries, never by transport chunk size.
Equivalent source ranks share a cumulative stream across parameters. Destination
DP, replicated parameter views and independent engines are all ordinary owners.
"""

import itertools
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass

from shardstream.plan import OverlapRegion, ShardOffset, _dtype_element_size, logger


@dataclass
class _Region:
    name: str
    start: tuple
    end: tuple
    sources: tuple
    destinations: tuple
    row_bytes: int
    alignment: int


def _covers(shard, start, end):
    return all(a <= x and y <= a + n for a, n, x, y in
               zip(shard.global_offset, shard.shape, start, end))


def _owners(meta):
    # A physical owner appears only once, even when a resolver includes it in
    # several replica views. Conflicting views of one parameter are invalid.
    owners = {}
    for shard in meta.shards:
        previous = owners.setdefault(int(shard.global_rank), shard)
        if (previous.global_offset, previous.shape, previous.dtype) != (
            shard.global_offset, shard.shape, shard.dtype
        ):
            raise ValueError(f"Conflicting physical ownership for {meta.name}")
    return tuple(owners[r] for r in sorted(owners))


def _regions(name, training, inference):
    if tuple(training.global_shape) != tuple(inference.global_shape):
        raise ValueError(f"Logical shape mismatch for {name}")
    sources, destinations = _owners(training), _owners(inference)
    shape = tuple(training.global_shape)
    axes = []
    for dim, size in enumerate(shape):
        cuts = {0, size}
        for shard in (*sources, *destinations):
            if len(shard.shape) != len(shape) or len(shard.global_offset) != len(shape):
                raise ValueError(f"Invalid ownership dimensionality for {name}")
            start, end = shard.global_offset[dim], shard.global_offset[dim] + shard.shape[dim]
            if start < 0 or end > size or start >= end:
                raise ValueError(f"Invalid ownership bounds for {name}")
            cuts.update((start, end))
        ordered = sorted(cuts)
        axes.append(tuple(zip(ordered, ordered[1:])))
    for cell in itertools.product(*axes):
        start, end = tuple(a for a, _ in cell), tuple(b for _, b in cell)
        receivers = tuple(s for s in destinations if _covers(s, start, end))
        if not receivers:
            continue
        candidates = tuple(s for s in sources if _covers(s, start, end))
        if not candidates:
            raise ValueError(f"Uncovered training region for {name}: {start} .. {end}")
        fp8 = str(inference.dtype).removeprefix("torch.") == "float8_e4m3fn"
        alignment = int(os.environ.get("SHARDSTREAM_FP8_BLOCK_ROWS", "128")) if fp8 else 1
        if fp8 and (len(shape) != 2 or any(
            a % block or b % block for a, b, block in
            zip(start, end, (alignment, int(os.environ.get("SHARDSTREAM_FP8_BLOCK_COLS", "128"))))
        )):
            raise ValueError(f"Logical ownership splits an FP8 quantization tile for {name}")
        yield _Region(name, start, end, candidates, receivers,
                      math.prod(b - a for a, b in zip(start[1:], end[1:]))
                      * _dtype_element_size(training.dtype), alignment)


def _source_partitions(sources):
    """Equal DP ranges, then equal physical-owner ranges within each DP.

    Dense and expert DP identities need not coincide. When DP identities do
    not describe a complete group, physical coverage still supplies every valid
    source, including TP-replicated scalars, router weights and CP-only copies.
    """
    sizes = {s.dp_size for s in sources}
    ids = {s.dp_rank for s in sources}
    if len(sizes) == 1 and ids == set(range(next(iter(sizes)))):
        groups = [[s for s in sources if s.dp_rank == dp] for dp in sorted(ids)]
    else:
        groups = [[s] for s in sources]
    result = []
    for group_index, group in enumerate(groups):
        group.sort(key=lambda s: (s.cp_rank, s.global_rank))
        for index, source in enumerate(group):
            denominator = len(groups) * len(group)
            result.append((source, group_index * len(group) + index,
                           group_index * len(group) + index + 1, denominator))
    return result


def build_logical_replica_operations(builder, inference, training, names, local_rank):
    groups = defaultdict(list)
    for name in names:
        for region in _regions(name, training[name], inference[name]):
            groups[tuple((s.global_rank, s.dp_rank, s.dp_size, s.cp_rank)
                         for s in region.sources)].append(region)

    operations, route_ids, loads = [], {}, defaultdict(int)
    region_count = splits = unique_bytes = destination_bytes = 0
    for key, regions in sorted(groups.items()):
        total = sum((r.end[0] - r.start[0]) * r.row_bytes for r in regions)
        partitions = _source_partitions(regions[0].sources)
        cursor = 0
        for region in regions:
            rows = region.end[0] - region.start[0]
            if builder.dp_source_policy == "stream":
                pieces = []
                for source, lo, hi, denominator in partitions:
                    def cut(numerator):
                        value = (total * numerator // denominator - cursor)
                        return max(0, min(rows, value // (region.row_bytes * region.alignment)
                                          * region.alignment))
                    begin, finish = cut(lo), cut(hi)
                    if finish > begin:
                        pieces.append((source, begin, finish))
            else:
                source = min(region.sources, key=lambda s: (loads[s.global_rank], s.global_rank))
                pieces = [(source, 0, rows)]
            if sum(end - start for _, start, end in pieces) != rows:
                raise ValueError(f"Incomplete source partition for {region.name}")
            cursor += rows * region.row_bytes
            region_count += 1
            splits += len(pieces) > 1
            for source, begin, finish in pieces:
                # A stream shares rank identities, not parameter metadata or
                # physical tensor geometry. Bind this region's actual owner.
                source = next(s for s in region.sources if s.global_rank == source.global_rank)
                start = (region.start[0] + begin, *region.start[1:])
                end = (region.start[0] + finish, *region.end[1:])
                nbytes = (finish - begin) * region.row_bytes
                loads[source.global_rank] += nbytes
                unique_bytes += nbytes
                source_rank = builder._compute_shard_transfer_rank(source, 0, False)
                destinations = sorted(
                    (builder._compute_shard_transfer_rank(shard, engine, True), shard)
                    for engine in range(builder.num_infer_engines)
                    for shard in region.destinations
                )
                targets = tuple(rank for rank, _ in destinations)
                if len(set(targets)) != len(targets):
                    raise ValueError(f"Duplicate physical receiver for {region.name}")
                route_key = (source_rank, targets)
                route_id = route_ids.setdefault(route_key, len(route_ids))
                source_offset = ShardOffset(source, source.global_offset,
                    tuple(a + b for a, b in zip(source.global_offset, source.shape)),
                    source.shape, source_rank)
                destination_bytes += nbytes * len(destinations)
                for rank, shard in destinations:
                    if local_rank is not None and local_rank not in (source_rank, rank):
                        continue
                    target_offset = ShardOffset(shard, shard.global_offset,
                        tuple(a + b for a, b in zip(shard.global_offset, shard.shape)),
                        shard.shape, rank)
                    op = builder._build_region_communication_plan(
                        OverlapRegion(target_offset, source_offset, start, end)
                    )[0]
                    op.replica_targets = targets
                    op.replica_route_id = route_id
                    operations.append(op)
    builder.logical_replica_stats = dict(
        source_groups=len(groups), ownership_regions=region_count, boundary_splits=splits,
        routes=len(route_ids), unique_source_bytes=unique_bytes,
        destination_bytes=destination_bytes, bytes_by_source=dict(loads),
    )
    logger.info("SHARDSTREAM_LOGICAL_REPLICA_PLAN " + json.dumps(builder.logical_replica_stats, sort_keys=True))
    return operations
