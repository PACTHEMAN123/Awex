"""Resumable four-backend matrix, global exclusive GPU lock, raw evidence audit."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from audit import analyze
from cleanup import cleanup_shm
from plan import IPS, matrix

BASE=Path('/tmp/figure11-1025'); ROOT=BASE/'results'; SOURCE=BASE/'source'; STAGE=BASE/'stage'
DEPS=f'/tmp/verl-nccl-1020-deps:/tmp/verl-nccl-1020-verl:{BASE}/awex:/tmp/comm-memory-1021-mooncake'

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

@ray.remote(num_cpus=0)
def inventory():
    return dict(ip=ray.util.get_node_ip_address(), captured=time.time(),
        prepared=json.loads((BASE/'prepared.json').read_text()),
        gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
        shm_files=[p.name for p in Path('/dev/shm').glob('nccl-*')],
        shm_free=os.statvfs('/dev/shm').f_bavail*os.statvfs('/dev/shm').f_frsize)

@ray.remote(num_cpus=0)
def collect(run_dir):
    p=Path(run_dir);assert p.is_relative_to(BASE/'runs')
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        if p.exists():
            for f in sorted(p.rglob('*')):
                if f.is_file():archive.add(f,arcname=str(f.relative_to(p)))
    return dict(ip=ray.util.get_node_ip_address(),data=stream.getvalue())

def node_calls(task,ips,*args):
    ray.init(address=IPS[0]+':6381',logging_level='ERROR')
    try:
        nodes={n['NodeManagerAddress']:n['NodeID'] for n in ray.nodes() if n['Alive']}
        assert all(ip in nodes for ip in ips),nodes
        arguments=[args] * len(ips)
        if task==cleanup_shm:arguments=[(row,) for row in args[0]]
        return ray.get([task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[ip],soft=False)).remote(*arg) for ip,arg in zip(ips,arguments)])
    finally:ray.shutdown()

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend');p.add_argument('--limit',type=int);a=p.parse_args()
    ROOT.mkdir(exist_ok=True)
    with open('/mnt/fuse/verl-e2e/runs/five-model-baselines-20261012/matrix/active.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        accepted=[];started=0
        for row in matrix():
            if a.backend and row['backend']!=a.backend:continue
            label=f"{row['case_id']}-{row['backend']}-r{row['repeat']}"
            complete=list(ROOT.glob(label+'-attempt*/complete.json'))
            if complete:
                assert len(complete)==1
                accepted.append(json.loads((complete[0].parent/'summary.json').read_text()));continue
            if a.limit is not None and started>=a.limit:break
            started+=1
            attempt=1+len(list(ROOT.glob(label+'-attempt*')))
            directory=ROOT/f'{label}-attempt{attempt}';directory.mkdir()
            before=node_calls(inventory,IPS);save(directory/'inventory.json',before)
            assert all(not r['gpu_processes'].strip() for r in before),before
            assert len({r['prepared']['revision'] for r in before})==1
            assert len({r['prepared']['awex_revision'] for r in before})==1
            cmd=[sys.executable,'-m','shardstream_benchmarks.recipe_runner','qwen3-30b-a3b',
                '--ray-address',IPS[0]+':6381','--training-hosts',*row['training_hosts'],
                '--rollout-hosts',*row['rollout_hosts'],'--model-path','/mnt/fuse/verl-e2e/models/Qwen3-30B-A3B',
                '--source',str(SOURCE),'--stage',str(STAGE),'--run-root',str(BASE/'runs'),
                '--output',str(directory/'outcome.json'),'--layout',row['layout'],
                '--rollout-replicas',str(row['replicas']),'--publication-mechanism',row['mechanism'],
                '--publication-bucket-mb','256','--model-weight-profile','--disable-prefix-caching',
                '--extra-python-path',DEPS,'--extra-library-path','/tmp/verl-nccl-1020-deps/nvidia/cuda_runtime/lib',
                '--runtime','/tmp/verl-nccl-1020-runtime','--nccl-lib','/tmp/verl-nccl-1020-nccl/nvidia/nccl/lib',
                '--num-updates','35','--warmup-updates','5','--timeout','3600',
                '--nccl-buffer-bytes','4194304','--dp-source-policy','stream',
                '--dp-source-ring-order','source-node','--replica-routing','logical']
            if row['backend']=='shardstream':
                cmd+=['--transport-gin-connections','4','--transport-max-channels','64',
                      '--transport-gin-fifo-depth','16','--transport-compact-fifo']
            case=dict(**row,label=label,attempt=attempt,revision=before[0]['prepared']['revision'],
                      awex_revision=before[0]['prepared']['awex_revision'],command=cmd)
            save(directory/'case.json',case)
            save(ROOT/'active.json',dict(state='running',directory=directory.name,started=time.time(),pid=os.getpid()))
            print(json.dumps(dict(event='start',label=label,directory=directory.name)),flush=True)
            env=dict(os.environ,PYTHONPATH=str(STAGE)+':'+DEPS,SHARDSTREAM_RECIPE_REPOSITORY=str(SOURCE))
            with (directory/'controller.log').open('w') as out:
                child=subprocess.Popen(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
                try:code=child.wait(timeout=3900)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:code=child.wait(timeout=180)
                    except subprocess.TimeoutExpired:child.kill();code=child.wait()
            save(directory/'exit.json',dict(code=code))
            if (directory/'outcome.json').exists():
                outcome=json.loads((directory/'outcome.json').read_text());manifest=[]
                for item in node_calls(collect,row['training_hosts']+row['rollout_hosts'],outcome['run_dir']):
                    name=item['ip']+'.tar.gz';(directory/name).write_bytes(item['data'])
                    manifest.append(dict(ip=item['ip'],file=name,bytes=len(item['data']),sha256=hashlib.sha256(item['data']).hexdigest()))
                save(directory/'manifest.json',manifest)
            after=node_calls(inventory,IPS);save(directory/'final-inventory.json',after)
            assert all(not r['gpu_processes'].strip() for r in after),'Run did not release GPUs'
            save(directory/'shm-cleanup.json',node_calls(cleanup_shm,IPS,before))
            try:
                result,network,metrics,records=analyze(directory)
                save(directory/'summary.json',result);save(directory/'network.json',network)
                save(directory/'native-metrics.json',metrics);save(directory/'records.json',records)
                save(directory/'complete.json',dict(passed=True,finished=time.time()))
                accepted.append(result)
            except Exception:
                save(directory/'audit-failed.json',dict(traceback=traceback.format_exc()))
                save(ROOT/'active.json',dict(state='needs-investigation',directory=directory.name,finished=time.time()))
                raise
            save(ROOT/'progress.json',dict(accepted=accepted,finished=time.time()))
            print(json.dumps(dict(event='passed',label=label,p50_ms=result['p50_ms'])),flush=True)
        state='complete' if len(accepted)==132 else 'qualification-complete'
        save(ROOT/'active.json',dict(state=state,passed=len(accepted),finished=time.time()))

if __name__=='__main__':main()
