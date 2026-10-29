"""Hold the experiment lock while exercising native CUDA RDMA on two hosts."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

BASE=Path('/tmp/figure11-1025')
STAGE=BASE/'stage'
DEPS=f'/tmp/verl-nccl-1020-deps:/tmp/verl-nccl-1020-verl:{BASE}/awex:/tmp/comm-memory-1021-mooncake'

@ray.remote(num_cpus=1,num_gpus=8)
def run(role):
    from shardstream.topology import _active_rdma_endpoints
    ip=ray.util.get_node_ip_address()
    devices=','.join(e.name for e in _active_rdma_endpoints())
    assert len(devices.split(','))==4,devices
    env=dict(os.environ,PYTHONPATH=str(STAGE)+':'+DEPS,
        AWEX_MOONCAKE_HOST=ip,AWEX_MOONCAKE_PROTOCOL='rdma',AWEX_MOONCAKE_DEVICE=devices,
        MC_GID_INDEX='3',AWEX_DEVICE_TYPE='cuda',
        LD_LIBRARY_PATH='/tmp/verl-nccl-1020-nccl/nvidia/nccl/lib:/usr/local/cuda/lib64:/tmp/verl-nccl-1020-deps/nvidia/cuda_runtime/lib:/tmp/verl-nccl-1020-runtime/lib/python3.12/site-packages/torch/lib')
    result=subprocess.run(['/tmp/verl-nccl-1020-runtime/bin/python','-u',
        str(BASE/'source/benchmarks/experiments/figure11_20261025/rdma_probe.py'),
        role,'--host','11.18.56.89'],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    (BASE/('rdma-probe-'+role+'.log')).write_text(result.stdout)
    return dict(ip=ip,role=role,code=result.returncode,log=result.stdout,devices=devices)

with open('/mnt/fuse/verl-e2e/runs/five-model-baselines-20261012/matrix/active.lock','a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ray.init(address='11.18.56.89:6381',logging_level='ERROR',runtime_env={'env_vars':{'PYTHONPATH':str(STAGE)+':'+DEPS}})
    nodes={n['NodeManagerAddress']:n['NodeID'] for n in ray.nodes() if n['Alive']}
    rows=ray.get([run.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[ip],soft=False)).remote(role)
                  for ip,role in [('11.18.56.89','send'),('11.18.50.148','receive')]])
    (BASE/'rdma-qualification.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        print(json.dumps(row),flush=True)
        assert row['code']==0,row
        events=[json.loads(line) for line in row['log'].splitlines() if line.startswith('{"role"')]
        assert len(events)==3 and all(e['verified'] and e['protocol']=='rdma' for e in events)
    ray.shutdown()
