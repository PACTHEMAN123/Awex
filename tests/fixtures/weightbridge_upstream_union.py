# Pinned WeightBridge 13587e8; unchanged BSD-style Meta reference.
from __future__ import annotations

def _shard_contained_in(inner: Shard, outer: Shard) -> bool:
    """True if *inner*'s box lies inside *outer* (same dimensionality and width metadata)."""
    if len(inner) != len(outer):
        return False
    for (li, ri, wi), (lo, ro, wo) in zip(inner, outer):
        if abs(wi) != abs(wo):
            return False
        if not (lo <= li < ri <= ro):
            return False
    return True

def _try_merge_two_shards(a: Shard, b: Shard) -> Shard | None:
    """If *a* and *b* match on all but one axis, and on that axis the intervals touch or overlap, return the union."""
    if len(a) != len(b):
        return None
    diff_dim: int | None = None
    for d, ((la, ra, wa), (lb, rb, wb)) in enumerate(zip(a, b)):
        if abs(wa) != abs(wb) or (wa < 0) != (wb < 0):
            return None
        if (la, ra) != (lb, rb):
            if diff_dim is not None:
                return None
            diff_dim = d
    if diff_dim is None:
        return None
    la, ra, w = a[diff_dim]
    lb, rb, _ = b[diff_dim]
    if ra < lb or rb < la:
        return None
    ml, mr = min(la, lb), max(ra, rb)
    if ml >= mr:
        return None
    out = list(a)
    out[diff_dim] = (ml, mr, w)
    return out

def _merge_shards_for_src_shard_spec(shards: Shards) -> Shards:
    """Collapse duplicate / contained shards and merge axis-aligned neighbors that differ on one axis only."""
    keep = [True] * len(shards)
    i = 0
    while i < len(keep):
        if not keep[i]:
            i += 1
            continue
        cur_shard = shards[i]
        for j in range(0, i):
            if keep[j]:
                shard = shards[j]
                if _shard_contained_in(shard, cur_shard):
                    keep[j] = False
                elif _shard_contained_in(cur_shard, shard):
                    keep[i] = False
                    break
                elif c := _try_merge_two_shards(shard, cur_shard):
                    shards.append(c)
                    keep.append(True)
                    keep[i] = False
                    keep[j] = False
                    break
        i += 1
    return [shard for i, shard in enumerate(shards) if keep[i]]
