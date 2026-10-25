"""Fresh native RDMA points in current paper Figures 9, 10 and 13."""
from plan import IPS

# Explicit axes from the frozen paper configurations. B is shared by Figs 9/10.
CASES=[
 ('A','qwen3-8b','Qwen3-8B','qwen8-tp2-pp2-r4tp2',4,1,(2,2,1,1,1),(2,1,1)),
 ('B','qwen3-30b-a3b','Qwen3-30B-A3B','qwen30-tp2-cp2-ep8-r4tp4',4,2,(2,1,2,8,1),(4,1,1)),
 ('C','gpt-oss-20b','gpt-oss-20b-BF16','gptoss16-r2tp4',2,1,(4,1,1,8,1),(4,1,1)),
 ('D','moonlight-16b-a3b','Moonlight-16B-A3B','moonlight-16gpu-tp4-ep8-r4tp4',4,2,(4,1,1,8,1),(4,1,1)),
 ('E','glm-4.7-flash','GLM-4.7-Flash','glm16-r1dp8ep8',1,1,(2,2,2,8,1),(1,8,8)),
 ('F','qwen3-30b-a3b','Qwen3-30B-A3B','qwen30-tp4-cp2-ep8-r4tp4',4,2,(4,1,2,8,1),(4,1,1)),
 ('G','qwen3-30b-a3b','Qwen3-30B-A3B','qwen30-tp2-pp4-cp2-ep4-r4tp4',4,2,(2,4,2,4,1),(4,1,1)),
 ('memory-default','qwen3-30b-a3b','Qwen3-30B-A3B','qwen30-train8-tp4-cp2-ep8-r1tp4',2,1,(4,1,2,8,1),(4,1,1)),
]

def matrix():
 rows=[]
 for config,recipe,model,layout,replicas,hosts,train,infer in CASES:
  memory=config=='memory-default';train_hosts=IPS[:1] if memory else IPS[:2]
  rollout_hosts=IPS[2:2+hosts]
  training=dict(zip(['tp','pp','cp','ep','etp'],train),world_size=8*len(train_hosts))
  for repeat in range(3):
   rows.append(dict(case_id='paper-'+config,config=config,recipe=recipe,model=model,
    series=['memory'] if memory else (['fig9','fig10'] if config=='B' else ['fig9' if config in 'ACDE' else 'fig10']),
    layout=layout,training_hosts=train_hosts,rollout_hosts=rollout_hosts,training=training,
    training_gpus=training['world_size'],rollout_gpus=replicas*infer[0]*infer[1],
    replicas=replicas,tp=infer[0],dp=infer[1],ep=infer[2],repeat=repeat,
    backend='awex',mechanism='awex_native_rdma',memory_profile=memory))
 assert len(rows)==24 and len({(r['case_id'],r['repeat']) for r in rows})==24
 return rows
