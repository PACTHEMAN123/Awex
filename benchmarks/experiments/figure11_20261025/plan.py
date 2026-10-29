"""Frozen Figure 11 matrix; roles never share GPUs or physical hosts."""
IPS=['11.18.56.89','11.18.50.148','11.18.50.220','11.17.132.185']
BACKENDS={'nccl':'verl_native_nccl','nixl':'verl_native_nixl','awex':'awex_native_nccl','shardstream':'shardstream'}

def cases(repeat):
 train=IPS[:2];infer=IPS[2:]
 rows=[]
 for dp,tp in [(1,8),(2,4),(4,2),(8,1)]:
  rows.append(dict(case_id=f'intra-dp{dp}',series=['intra']+(['sender-dp2'] if dp==2 else []),
   layout='qwen30-tp4-cp2-ep8-r2tp8ep8' if dp==1 else f'qwen30-tp4-cp2-ep8-r2tp{tp}dp{dp}ep8',
   training_hosts=train,rollout_hosts=infer,replicas=2,training_gpus=16,rollout_gpus=16,tp=tp,dp=dp,ep=8,trainer_dp=2))
 pool=IPS[1:];rotated=pool[repeat:]+pool[:repeat]
 for n in range(1,7):
  rows.append(dict(case_id=f'inter-r{n}',series=['inter'],layout='qwen30-train8-tp4-cp2-ep8-r1tp4',
   training_hosts=train[:1],rollout_hosts=rotated[:min(n,3)],replicas=n,training_gpus=8,
   rollout_gpus=n*4,tp=4,dp=1,ep=1,trainer_dp=1))
 rows.append(dict(case_id='sender-dp1',series=['sender-dp1'],layout='qwen30-train8-tp4-cp2-ep8-r2tp4dp2ep8',
  training_hosts=train[:1],rollout_hosts=infer,replicas=2,training_gpus=8,rollout_gpus=16,tp=4,dp=2,ep=8,trainer_dp=1))
 return rows

def matrix():
 rows=[]
 for repeat in range(3):
  for index,case in enumerate(cases(repeat)):
   keys=list(BACKENDS);offset=(index+repeat)%4;keys=keys[offset:]+keys[:offset]
   for backend in keys:rows.append(dict(**case,repeat=repeat,backend=backend,mechanism=BACKENDS[backend]))
 # Inspect the smallest and largest deployments early, using their actual formal launches.
 priority={'inter-r1':0,'intra-dp8':1,'inter-r6':2}
 rows.sort(key=lambda r:(0 if r['repeat']==0 and r['case_id'] in priority else 1,priority.get(r['case_id'],9) if r['repeat']==0 else 9))
 assert len(rows)==132 and len({(r['case_id'],r['repeat'],r['backend']) for r in rows})==132
 assert sum(r['rollout_gpus'] for r in rows)==1968
 return rows


def matching_completions(root, row):
 """A completed run from another transport never satisfies the active plan."""
 import json
 label=f"{row['case_id']}-{row['backend']}-r{row['repeat']}"
 return [p for p in root.glob(label+'-attempt*/complete.json')
         if json.loads(p.with_name('case.json').read_text())['mechanism']==row['mechanism']]
