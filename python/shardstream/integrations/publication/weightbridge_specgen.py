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


class _Intervals:
    """Dynamic inclusive overlap index; priority never affects merge ordering."""
    def __init__(self):
        import random
        self.root = None
        self.rng = random.Random(20261022)

    @staticmethod
    def refresh(node):
        if node is not None:
            node[3] = max(node[1], node[4][3] if node[4] else -math.inf,
                          node[5][3] if node[5] else -math.inf)
        return node

    def join(self, left, right):
        if left is None: return right
        if right is None: return left
        if left[2] < right[2]:
            left[5] = self.join(left[5], right)
            return self.refresh(left)
        right[4] = self.join(left, right[4])
        return self.refresh(right)

    def put(self, root, node):
        if root is None: return node
        branch = 4 if node[0] < root[0] else 5
        root[branch] = self.put(root[branch], node)
        child = root[branch]
        if child[2] < root[2]:
            other = 5 if branch == 4 else 4
            root[branch] = child[other]
            child[other] = self.refresh(root)
            return self.refresh(child)
        return self.refresh(root)

    def drop(self, root, key):
        assert root is not None
        if root[0] == key: return self.join(root[4], root[5])
        branch = 4 if key < root[0] else 5
        root[branch] = self.drop(root[branch], key)
        return self.refresh(root)

    def add(self, left, right, index):
        node = [(left, index), right, self.rng.getrandbits(64), right, None, None]
        self.root = self.put(self.root, node)

    def remove(self, left, index):
        self.root = self.drop(self.root, (left, index))

    def overlaps(self, left, right):
        result = []
        def visit(node):
            if node is None or node[3] < left: return
            visit(node[4])
            if node[0][0] > right: return
            if node[1] >= left: result.append(node[0][1])
            visit(node[5])
        visit(self.root)
        return result


def indexed_source_union(shards, contained, merge):
    """Preserve upstream source-union first-match order and list mutation.

    Every containing or mergeable box must overlap (or touch) on any selected
    axis. An interval query removes only impossible prior candidates; native
    predicates still decide every containment and merge in original j order.
    """
    keep = [True] * len(shards)
    indexes = {}
    def group(shard): return tuple(abs(width) for _, _, width in shard)
    def register(index, remove=False):
        shard = shards[index];key = group(shard)
        tables = indexes.get(key)
        if tables is None:
            tables = [_Intervals() for _ in shard]
            indexes[key] = tables
        for table, (left, right, _) in zip(tables, shard):
            if remove: table.remove(left, index)
            else: table.add(left, right, index)
    i = 0
    while i < len(keep):
        if not keep[i]: i += 1;continue
        current = shards[i];key = group(current)
        tables = indexes.get(key)
        if current and tables:
            axis = min(range(len(current)), key=lambda a: (current[a][1]-current[a][0])/max(1, abs(current[a][2])))
            left, right, _ = current[axis]
            candidates = sorted(tables[axis].overlaps(left, right))
        elif not current:
            # Scalar source boxes do not occur in the native source planner.
            candidates = [j for j in range(i) if keep[j] and not shards[j]]
        else: candidates = []
        for j in candidates:
            previous = shards[j]
            if contained(previous, current):
                keep[j] = False;register(j, remove=True)
            elif contained(current, previous):
                keep[i] = False;break
            else:
                combined = merge(previous, current)
                if combined:
                    shards.append(combined);keep.append(True)
                    keep[i] = False;keep[j] = False;register(j, remove=True);break
        if keep[i]: register(i)
        i += 1
    return [shard for index, shard in enumerate(shards) if keep[index]]
