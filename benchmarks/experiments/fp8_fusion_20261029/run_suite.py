"""Small paired same-wire FP8 fusion experiment; owns only this run's processes."""
import fcntl,hashlib,io,json,os,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from evidence import analyze,audit_affinity,cleanup_shm
BASE=Path('/tmp/fp8-fusion-resume-1029');ROOT=BASE/'results'
ASSETS=Path('/tmp/fp8-fusion-1029');SOURCE=ASSETS/'source';STAGE=ASSETS/'stage'
IPS=['11.18.56.89','11.18.50.220','11.18.50.148','11.17.132.185']
DEPS='/tmp/verl-nccl-1020-deps:/tmp/verl-nccl-1020-verl:/tmp/figure11-1025/awex'
REV=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
@ray.remote(num_cpus=0)
def inventory():
 p=Path('/tmp/fp8-fusion-1029')
 prepared=json.loads((p/'prepared.json').read_text())
 native=list((p/'stage/shardstream').glob('*.so'))
 assert len(native)==1
 assert hashlib.sha256(native[0].read_bytes()).hexdigest()==prepared['native_sha256']
 assert prepared['native_sha256']=='6de1b04659d22ebe6972f26d76344714eb11b61cd75bc1b3b5b2fcb036c21d0a'
 assert prepared['revision']=='67c9cfeebc5f593fbd0e8f63001c2d64e766872a'
 assert not subprocess.check_output(['git','-C',str(p/'source'),'status','--porcelain'],text=True).strip()
 return dict(ip=ray.util.get_node_ip_address(),revision=subprocess.check_output(['git','-C',str(p/'source'),'rev-parse','HEAD'],text=True).strip(),prepared=json.loads((p/'prepared.json').read_text()),nccl=json.loads((p/'nccl/manifest.json').read_text()),nccl_sha256=hashlib.sha256((p/'nccl/lib/libnccl.so.2').read_bytes()).hexdigest(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),shm_files=[f.name for f in Path('/dev/shm').glob('nccl-*')],captured=time.time())
@ray.remote(num_cpus=0)
def collect(run_dir):
 p=Path(run_dir);assert p.is_relative_to(BASE/'runs')
 stream=io.BytesIO()
 with tarfile.open(fileobj=stream,mode='w:gz') as archive:
  if p.exists():
   for f in sorted(p.rglob('*')):
    if f.is_file():archive.add(f,arcname=str(f.relative_to(p)))
 return dict(ip=ray.util.get_node_ip_address(),data=stream.getvalue())
def calls(function,arguments=None):
 ray.init(address=IPS[0]+':6381',logging_level='ERROR',runtime_env={'env_vars':{'PYTHONPATH':str(BASE)+':'+str(STAGE)+':'+DEPS}})
 try:
  nodes={n['NodeManagerAddress']:n['NodeID'] for n in ray.nodes() if n['Alive']}
  assert set(IPS)<=set(nodes),nodes
  return ray.get([function.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[ip],soft=False)).remote(*(arguments[i] if arguments else ())) for i,ip in enumerate(IPS)])
 finally:ray.shutdown()
def mirror():
 target=Path('/mnt/fuse/verl-e2e/runs/shardstream-fp8-fusion-resume-20261029')
 shutil.copytree(ROOT,target,dirs_exist_ok=True)
def run(replicas,mode,repeat):
 label=f'tp2x{replicas}-{mode}-r{repeat}'
 completed=[]
 for previous in ROOT.glob(label+'*'):
  if (previous/'exit_code').exists() and (previous/'exit_code').read_text().strip()=='0' and (previous/'manifest.json').exists() and not (previous/'summary.json').exists():
   outcome=json.loads((previous/'outcome.json').read_text());audit_affinity(previous,outcome)
   result=analyze(previous)
   result.update(mode=mode,replicas=replicas,native_sha256=json.loads((previous/'inventory.json').read_text())[0]['prepared']['native_sha256'])
   save(previous/'summary.json',result)
  if (previous/'summary.json').exists() and (previous/'exit_code').read_text().strip()=='0':completed.append(previous)
 if completed:
  assert len(completed)==1
  result=analyze(completed[0]);result.update(mode=mode,replicas=replicas,native_sha256=json.loads((completed[0]/'inventory.json').read_text())[0]['prepared']['native_sha256']);return result
 directory=ROOT/label
 if directory.exists():directory=ROOT/(label+f'-attempt{len(list(ROOT.glob(label+"*")))+1}')
 directory.mkdir()
 before=calls(inventory);save(directory/'inventory.json',before)
 assert all(not r['processes'] and r['revision']==REV for r in before),before
 assert len({r['prepared']['native_sha256'] for r in before})==1
 assert len({r['nccl_sha256'] for r in before})==1 and all(r['nccl']['version']==23102 and r['nccl']['files']['libnccl.so.2']==r['nccl_sha256'] for r in before)
 cmd=[sys.executable,'-m','shardstream_benchmarks.recipe_runner','qwen3-30b-a3b','--ray-address',IPS[0]+':6381',
 '--training-hosts',*IPS[2:],'--rollout-hosts',*IPS[:2],'--model-path','/mnt/fuse/verl-e2e/models/Qwen3-30B-A3B',
 '--source',str(SOURCE),'--stage',str(STAGE),'--run-root',str(BASE/'runs'),'--output',str(directory/'outcome.json'),
 '--layout','qwen30-tp2-cp2-ep8-r8tp2','--rollout-replicas',str(replicas),'--gpu-placement','nic-spread',
 '--publication-mechanism','shardstream','--model-weight-profile','--weight-format','fp8','--fp8-execution',mode,
 '--disable-prefix-caching','--rollout-gpu-memory-utilization','0.35','--extra-python-path',DEPS,'--extra-library-path','/tmp/verl-nccl-1020-deps/nvidia/cuda_runtime/lib',
 '--runtime','/tmp/verl-nccl-1020-runtime','--nccl-lib','/tmp/fp8-fusion-1029/nccl/lib',
 '--num-updates','20','--warmup-updates','5','--timeout','2400','--dp-source-policy','stream','--dp-source-ring-order','source-node','--replica-routing','logical',
 '--transport-gin-connections','4','--transport-max-channels','64','--transport-gin-fifo-depth','16','--transport-compact-fifo','--fifo-layout-profile']
 case=dict(case_id=label,series=['fp8-fusion'],revision=REV,repeat=repeat,routing='logical',mode=mode,replicas=replicas,command=cmd)
 save(directory/'case.json',case);save(ROOT/'active.json',dict(state='running',case=label,started=time.time(),pid=os.getpid()))
 print(json.dumps(dict(event='start',case=label)),flush=True)
 env=dict(os.environ,PYTHONPATH=str(STAGE)+':'+DEPS,SHARDSTREAM_RECIPE_REPOSITORY=str(SOURCE))
 with (directory/'controller.log').open('w') as out:
  child=subprocess.Popen(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
  try:code=child.wait(timeout=3000)
  except subprocess.TimeoutExpired:
   child.terminate()
   try:code=child.wait(timeout=180)
   except subprocess.TimeoutExpired:child.kill();code=child.wait()
 (directory/'exit_code').write_text(str(code)+'\n')
 if (directory/'outcome.json').exists():
  outcome=json.loads((directory/'outcome.json').read_text());manifest=[]
  for r in calls(collect,[(outcome['run_dir'],)]*4):
   name=r['ip']+'.tar.gz';(directory/name).write_bytes(r['data']);manifest.append(dict(ip=r['ip'],file=name,bytes=len(r['data']),sha256=hashlib.sha256(r['data']).hexdigest()))
  save(directory/'manifest.json',manifest)
 after=calls(inventory)
 for _ in range(12):
  if all(not r['processes'] for r in after):break
  time.sleep(5);after=calls(inventory)
 save(directory/'final-inventory.json',after)
 assert all(not r['processes'] for r in after),after
 save(directory/'shm-cleanup.json',calls(cleanup_shm,[(r,) for r in before]))
 mirror()
 assert code==0,(label,code)
 audit_affinity(directory,outcome)
 result=analyze(directory)
 result.update(mode=mode,replicas=replicas,native_sha256=before[0]['prepared']['native_sha256'])
 save(directory/'summary.json',result);mirror()
 print(json.dumps(dict(event='passed',case=label,p50_ms=result['p50_ms'])),flush=True)
 return result
ROOT.mkdir(parents=True,exist_ok=True)
lock=open('/mnt/fuse/verl-e2e/runs/five-model-baselines-20261012/matrix/active.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 before=calls(inventory);assert all(not r['processes'] for r in before),before
 result=subprocess.check_output([str(ASSETS/'fp8_staged')],text=True)
 save(ROOT/'wire-equivalence.json',json.loads(result))
 qualification=json.loads((ASSETS/'results/runtime-rebuild.json').read_text())
 assert qualification['passed']
 save(ROOT/'prior-runtime-qualification.json',qualification)
 results=[]
 # Three independent pairs, reversing the middle pair's order.
 replicas=4
 for repeat,modes in enumerate([['fused','staged'],['staged','fused'],['fused','staged']]):
  for mode in modes:
   if (BASE/'STOP_AFTER_CASE').exists():
    save(ROOT/'active.json',dict(state='stopped',finished=time.time()));mirror();sys.exit(0)
   results.append(run(replicas,mode,repeat));save(ROOT/'results.json',results);mirror()
 pairs=[]
 for repeat in range(3):
  fused=next(r for r in results if r['repeat']==repeat and r['mode']=='fused')
  staged=next(r for r in results if r['repeat']==repeat and r['mode']=='staged')
  for key in ['native_sha256','writer_bytes','fp8_wire_bytes','lowered','relay_chains','training','rollout']:
   assert fused[key]==staged[key],(replicas,key)
  fingerprints=[json.loads((ROOT/r['directory']/'transport-fingerprints.json').read_text()) for r in [fused,staged]]
  assert fingerprints[0]==fingerprints[1],('transport plan changed',replicas)
  pairs.append(dict(replicas=replicas,repeat=repeat,fused_p50_ms=fused['p50_ms'],staged_p50_ms=staged['p50_ms'],staged_over_fused=staged['p50_ms']/fused['p50_ms'],fusion_reduction_percent=(1-fused['p50_ms']/staged['p50_ms'])*100,same_wire_routes_fifo=True))
 save(ROOT/'paired-comparison.json',pairs)
 final=calls(inventory);assert all(not r['processes'] for r in final)
 save(ROOT/'final-audit.json',final)
 # Include shared-memory files left by preserved failed model attempts, using
 # each node's oldest task snapshot and refusing files referenced by processes.
 oldest={}
 for path in ROOT.glob('tp2x*/inventory.json'):
  for row in json.loads(path.read_text()):
   if row['ip'] not in oldest or row['captured']<oldest[row['ip']]['captured']:oldest[row['ip']]=row
 if oldest:
  cleaned=calls(cleanup_shm,[(oldest[ip],) for ip in IPS]);save(ROOT/'final-shm-cleanup.json',cleaned)

 save(ROOT/'active.json',dict(state='complete',launches=6,verified_updates=120,measured_updates=90,finished=time.time()));mirror()
except BaseException:
 save(ROOT/'active.json',dict(state='failed',traceback=traceback.format_exc(),finished=time.time()));mirror();raise
finally:lock.close()
