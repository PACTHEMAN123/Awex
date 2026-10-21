"""Equivalent ordering-preserving candidate index for WeightBridge specgen.

Only metadata coalescing is accelerated. Geometry, first-pair merge ordering,
transpose inference, sorting and runtime routes remain defined by upstream.
"""
import heapq
import math


def indexed_merge(mappings, adjacent):
    pending = {}
    seen = set()
    for source, destination in mappings:
        key = (tuple(source), tuple(destination))
        if key not in seen:
            seen.add(key)
            pending[len(pending)] = (list(source), list(destination))

    def keys(shard):
        for axis, (left, right, width) in enumerate(shard):
            signature = (axis, width, tuple(shard[:axis] + shard[axis + 1:]))
            yield signature + (left,), signature + (right,)

    while True:
        # A stable original list index defines upstream's first (i, j) pair.
        indexes = [({}, {}), ({}, {})]
        revisions = {index: 0 for index in pending}
        candidates = []

        def register(index, remove=False):
            for side, shard in enumerate(pending[index]):
                for starts, ends in keys(shard):
                    for table, key in zip(indexes[side], (starts, ends)):
                        if remove:
                            table[key].remove(index)
                            if not table[key]: del table[key]
                        else:
                            table.setdefault(key, set()).add(index)

        def neighbors(index, side):
            result = set()
            starts, ends = indexes[side]
            for left, right in keys(pending[index][side]):
                result.update(ends.get(left, ()))
                result.update(starts.get(right, ()))
            result.discard(index)
            return result

        def merged_pair(i, j):
            source = adjacent(pending[i][0], pending[j][0])
            destination = adjacent(pending[i][1], pending[j][1])
            if source is None or destination is None:
                return None
            if source == pending[i][0] or destination == pending[i][1]:
                return None
            if math.prod(r-l for l, r, _ in source) != math.prod(r-l for l, r, _ in destination):
                return None
            return source, destination

        def enqueue(index, initial=False):
            for other in neighbors(index, 0) & neighbors(index, 1):
                if initial and other < index: continue
                i, j = sorted((index, other))
                if merged_pair(i, j) is not None:
                    heapq.heappush(candidates, (i, j, revisions[i], revisions[j]))

        for index in pending: register(index)
        for index in pending: enqueue(index, initial=True)
        while candidates:
            i, j, ri, rj = heapq.heappop(candidates)
            if i not in pending or j not in pending or revisions[i] != ri or revisions[j] != rj:
                continue
            merged = merged_pair(i, j)
            assert merged is not None
            register(i, remove=True); register(j, remove=True)
            pending[i] = merged
            del pending[j]
            revisions[i] += 1
            register(i)
            enqueue(i)

        orientation_changed = False
        for index, (source, destination) in pending.items():
            if len(source) == len(destination) == 2 and all(width > 0 for _, _, width in source):
                source_shape = tuple(right-left for left, right, _ in source)
                destination_shape = tuple(right-left for left, right, _ in destination)
                if source_shape != destination_shape and source_shape[::-1] == destination_shape:
                    source = list(source)
                    left, right, width = source[0]
                    source[0] = (left, right, -abs(width))
                    pending[index] = (source, destination)
                    orientation_changed = True
        if not orientation_changed: break
    return sorted(pending.values(), key=lambda mapping: (tuple(mapping[0]), tuple(mapping[1])))
