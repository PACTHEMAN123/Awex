# Pinned WeightBridge 13587e8: unchanged reference for differential tests.
# Copyright Meta Platforms, Inc.; BSD-style upstream license.
from __future__ import annotations
import math

def _merge_adjacent_shards(a: Shard, b: Shard) -> Shard | None:
    """Merge boxes that are identical except for one exactly adjacent axis."""
    if len(a) != len(b):
        return None
    differing: int | None = None
    for axis, ((la, ra, wa), (lb, rb, wb)) in enumerate(zip(a, b)):
        if wa != wb:
            return None
        if (la, ra) != (lb, rb):
            if differing is not None:
                return None
            differing = axis
    if differing is None:
        return list(a)
    la, ra, width = a[differing]
    lb, rb, _ = b[differing]
    if ra != lb and rb != la:
        return None
    merged = list(a)
    merged[differing] = (min(la, lb), max(ra, rb), width)
    return merged

def _merge_consecutive_mappings(mappings: list[ShardMapping]) -> list[ShardMapping]:
    """Coalesce consecutive source and destination boxes where both agree."""
    pending: list[ShardMapping] = []
    seen: set[tuple] = set()
    for source, destination in mappings:
        key = (tuple(source), tuple(destination))
        if key not in seen:
            seen.add(key)
            pending.append((list(source), list(destination)))

    while True:
        changed = True
        while changed:
            changed = False
            for i in range(len(pending)):
                for j in range(i + 1, len(pending)):
                    source = _merge_adjacent_shards(pending[i][0], pending[j][0])
                    destination = _merge_adjacent_shards(pending[i][1], pending[j][1])
                    if source is None or destination is None:
                        continue
                    # Identical source or destination boxes represent replication,
                    # not consecutive shards, and must remain separate.
                    if source == pending[i][0] or destination == pending[i][1]:
                        continue
                    if math.prod(r - l for l, r, _ in source) != math.prod(
                        r - l for l, r, _ in destination
                    ):
                        continue
                    pending[i] = (source, destination)
                    pending.pop(j)
                    changed = True
                    break
                if changed:
                    break

        # A one-element-wide logical remainder can make transpose impossible
        # to identify during local extraction.  Once consecutive 1x1 boxes
        # have coalesced, their rectangular shapes make it unambiguous.
        orientation_changed = False
        for index, (source, destination) in enumerate(pending):
            if len(source) == len(destination) == 2 and all(
                width > 0 for _left, _right, width in source
            ):
                source_shape = tuple(right - left for left, right, _width in source)
                destination_shape = tuple(
                    right - left for left, right, _width in destination
                )
                if (
                    source_shape != destination_shape
                    and source_shape[::-1] == destination_shape
                ):
                    source = list(source)
                    left, right, width = source[0]
                    source[0] = (left, right, -abs(width))
                    pending[index] = (source, destination)
                    orientation_changed = True

        if not orientation_changed:
            break

    pending.sort(key=lambda mapping: (tuple(mapping[0]), tuple(mapping[1])))
    return pending
