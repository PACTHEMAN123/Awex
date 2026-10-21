"""Differential checks against exact pinned upstream's slow metadata routine."""
import ast,json,math,random,tarfile,time,__future__
from pathlib import Path
import importlib.util,runpy
root=Path(__file__).resolve().parents[1]
p=root/'python/shardstream/integrations/publication/weightbridge_specgen.py'
spec=importlib.util.spec_from_file_location('indexed_weightbridge',p)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
indexed_merge=module.indexed_merge
ns=runpy.run_path(str(root/'tests/fixtures/weightbridge_upstream_merge.py'))
slow=ns['_merge_consecutive_mappings'];adj=ns['_merge_adjacent_shards']
rng=random.Random(20261022);checked=0

def compare(case):
    global checked
    expected=slow(case);actual=indexed_merge(case,adj)
    assert actual==expected,(case,expected,actual)
    checked+=1

# Full grids under transposes, dimensional collapse, interleaving, replication,
# permutations, duplicates and fragments with non-unit extents.
for trial in range(1500):
    rows=rng.randint(1,5);cols=rng.randint(1,6)
    style=trial%6;parts=[]
    for r in range(rows):
        for c in range(cols):
            src=[(r,r+1,rows),(c,c+1,cols)]
            if style==0:dst=[(r,r+1,rows),(c,c+1,cols)]
            elif style==1:dst=[(c,c+1,cols),(r,r+1,rows)]
            elif style==2:dst=[(r*cols+c,r*cols+c+1,rows*cols)]
            elif style==3:src=[(r,r+1,rows),(c*2,c*2+1,cols*2)];dst=[(r,r+1,rows),(c,c+1,cols)]
            elif style==4:src=[(r,r+1,-rows),(c,c+1,cols)];dst=[(c,c+1,cols),(r,r+1,rows)]
            else:dst=[(r//2,r//2+1,rows),(c,c+1,cols)]
            if rng.random()<.9:parts.append((src,dst))
    if parts:parts.extend(rng.choices(parts,k=rng.randint(0,5)))
    rng.shuffle(parts);compare(parts)
# Arbitrary 1D/2D/3D boxes, invalid areas, zero extents and negative widths.
for trial in range(1500):
    parts=[]
    for _ in range(rng.randint(0,30)):
        pair=[]
        for side in range(2):
            shard=[]
            for axis in range(rng.randint(1,3)):
                left=rng.randint(0,3);right=left+rng.randint(0,3)
                shard.append((left,right,rng.choice([-8,8])))
            pair.append(shard)
        parts.append(tuple(pair))
    compare(parts)
# Actual problematic bias geometry: even/odd half interleave, no legal merge.
for n in (8,64,256):
    case=[([(2,3,32),(2*c,2*c+1,2*n)],[(c,c+1,2*n)]) for c in range(n)]
    compare(case)
n=5760
case=[([(2,3,32),(2*c,2*c+1,2*n)],[(c,c+1,2*n)]) for c in range(n)]
start=time.monotonic();result=indexed_merge(case,adj);elapsed=time.monotonic()-start
assert result==sorted(case,key=lambda m:(tuple(m[0]),tuple(m[1])))
report=dict(differential_cases=checked,large_interleaved_bias_fragments=n,large_case_seconds=elapsed,large_case_exact_no_merge=True)

print(json.dumps(report))

def test_indexed_matches_pinned_upstream():
    assert checked == 3003


def test_source_union_matches_upstream_candidates_and_input_mutation():
    import copy
    upstream=runpy.run_path(str(root/'tests/fixtures/weightbridge_upstream_union.py'))
    native=upstream['_merge_shards_for_src_shard_spec']
    contained=upstream['_shard_contained_in'];merge=upstream['_try_merge_two_shards']
    randomizer=random.Random(20261023)
    for trial in range(3000):
        dimensions=randomizer.randint(1,3)
        boxes=[]
        widths=[randomizer.randint(2,8) for _ in range(dimensions)]
        for _ in range(randomizer.randint(0,35)):
            box=[]
            for width in widths:
                left=randomizer.randint(0,width-1)
                right=randomizer.randint(left,width)
                box.append((left,right,randomizer.choice([width,-width])))
            boxes.append(box)
        original=copy.deepcopy(boxes);indexed=copy.deepcopy(boxes)
        expected=native(original)
        actual=module.indexed_source_union(indexed,contained,merge)
        assert actual==expected,(boxes,expected,actual)
        assert indexed==original
    for n in (16,64,256):
        boxes=[[(2,3,32),(c,c+1,n)] for c in list(range(0,n,2))+list(range(1,n,2))]
        original=copy.deepcopy(boxes);indexed=copy.deepcopy(boxes)
        assert module.indexed_source_union(indexed,contained,merge)==native(original)
        assert indexed==original
    n=5760
    boxes=[[(2,3,32),(c,c+1,n)] for c in list(range(0,n,2))+list(range(1,n,2))]
    actual=module.indexed_source_union(boxes,contained,merge)
    assert actual==[[(2,3,32),(0,n,n)]]
