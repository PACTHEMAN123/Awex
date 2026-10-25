"""Verify required checkpoint assets between the two GPU experiment suites."""
import hashlib
import json
from pathlib import Path
import time

import ray
from plan import IPS
from run_matrix import BASE, node_calls, save

@ray.remote(num_cpus=0)
def verify(rows):
 ip=ray.util.get_node_ip_address();checked=[]
 # The fourth node hosts only Qwen30 and Moonlight in the paper matrix.
 required={'Qwen3-30B-A3B','Moonlight-16B-A3B'} if ip==IPS[3] else None
 for row in rows:
  if required is not None and Path(row['path']).parts[1] not in required:continue
  p=Path('/mnt/fuse/verl-e2e')/row['path'];assert p.stat().st_size==row['bytes'],str(p)
  digest=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(16<<20),b''):digest.update(block)
  assert digest.hexdigest()==row['sha256'],str(p)
  checked.append(row)
 return dict(ip=ip,files=checked,verified_at=time.time())

if __name__=='__main__':
 rows=json.loads(Path(__file__).with_name('checkpoint-manifest.json').read_text())
 results=node_calls(verify,IPS,rows)
 save(BASE/'other-model-assets-verified.json',results)
 print(json.dumps({'checkpoint_verification_passed':True,'files_per_node':{r['ip']:len(r['files']) for r in results}}))
