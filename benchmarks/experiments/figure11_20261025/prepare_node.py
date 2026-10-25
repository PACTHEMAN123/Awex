"""Install committed source bundles and reuse a verified identical native build."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

BASE = Path('/tmp/figure11-1025')
SOURCE = BASE / 'source'
AWEX = BASE / 'awex'
STAGE = BASE / 'stage'
RUNTIME = Path('/tmp/verl-nccl-1020-runtime')
for target, bundle, branch in [
    (SOURCE, BASE/'shardstream.bundle', 'codex/figure11-four-backends-20261025'),
    (AWEX, BASE/'awex.bundle', 'codex/awex-rdma-model-adapters-20261025'),
]:
    if not target.exists():
        subprocess.run(['git', 'clone', '-b', branch, str(bundle), str(target)], check=True)
    else:
        subprocess.run(['git', '-C', str(target), 'fetch', str(bundle), branch], check=True)
        subprocess.run(['git', '-C', str(target), 'merge', '--ff-only', 'FETCH_HEAD'], check=True)
if not STAGE.exists():
    shutil.copytree('/tmp/weightbridge-fig9-1022/stage', STAGE, symlinks=True)
for name, original in [('shardstream', SOURCE/'python/shardstream'),
                       ('shardstream_benchmarks', SOURCE/'benchmarks/shardstream_benchmarks')]:
    target = STAGE/name
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    shutil.copytree(original, target, ignore=shutil.ignore_patterns('__pycache__'))
# copytree above replaces Python; keep the native extension from the same C++
# source. Its checked hash is the binary used by the verified compact-FIFO run.
previous = Path('/tmp/weightbridge-fig9-1022/stage')
native = next(previous.rglob('_native*.so'))
assert hashlib.sha256(native.read_bytes()).hexdigest() == 'eec00c08c7f9a44abbb1e4ead3da9ffa82419d42eecdc2bbce8d706cef3ad30a'
destination = STAGE/native.relative_to(previous)
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(native, destination)
revision = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
awex_revision = subprocess.check_output(['git','-C',str(AWEX),'rev-parse','HEAD'],text=True).strip()
assert awex_revision == 'f7ed45cc9b95fd62b7a78177da2dfaf2ee85cf75'
subprocess.run(['git','-C',str(AWEX),'merge-base','--is-ancestor',
                'bae159256a768d54465ed693d90d925025032a17','HEAD'],check=True)
subprocess.run(['git','-C',str(AWEX),'diff','--exit-code',
                'bae159256a768d54465ed693d90d925025032a17','HEAD','--',
                'awex/transfer','awex/writer/mooncake_writer.py',
                'awex/reader/mooncake_reader.py','awex/writer/astate_writer.py',
                'awex/reader/astate_reader.py'],check=True)
env = dict(os.environ,
    PYTHONPATH=f'{STAGE}:/tmp/verl-nccl-1020-deps:/tmp/verl-nccl-1020-verl:{AWEX}:/tmp/comm-memory-1021-mooncake',
    LD_LIBRARY_PATH='/tmp/verl-nccl-1020-nccl/nvidia/nccl/lib:/usr/local/cuda/lib64:/tmp/verl-nccl-1020-deps/nvidia/cuda_runtime/lib:'+str(RUNTIME/'lib/python3.12/site-packages/torch/lib'))
code = '''import json,torch,ray,vllm,mooncake.engine,awex.transfer.mooncake_comm
from importlib.metadata import version
from pathlib import Path
import hashlib
print(json.dumps(dict(torch=torch.__version__,ray=ray.__version__,vllm=vllm.__version__,mooncake=mooncake.engine.__file__,mooncake_version=version('mooncake-transfer-engine'),mooncake_sha256=hashlib.sha256(Path(mooncake.engine.__file__).read_bytes()).hexdigest(),awex=awex.transfer.mooncake_comm.__file__)))'''
imports = subprocess.check_output([str(RUNTIME/'bin/python'),'-c',code],env=env,text=True)
dependency=json.loads(imports);assert dependency['mooncake_version']=='0.3.13.post1'
row = dict(hostname=socket.gethostname(), revision=revision, awex_revision=awex_revision,
           native_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
           awex_transport_sha256=hashlib.sha256((AWEX/'awex/transfer/mooncake_comm.py').read_bytes()).hexdigest(),
           mooncake_version=dependency['mooncake_version'],mooncake_sha256=dependency['mooncake_sha256'],
           imports=imports,
           gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True))
assert not row['gpu_processes'].strip(), row['gpu_processes']
artifact = BASE/'astate-artifact.json'
if artifact.exists():
    manifest=json.loads(artifact.read_text())
    for name, checksum in manifest['files'].items():
        assert hashlib.sha256((BASE/'astate-runtime'/name).read_bytes()).hexdigest()==checksum
    astate_env=dict(env, PYTHONPATH=manifest['pythonpath'], LD_LIBRARY_PATH=manifest['ld_library_path'])
    row['astate_imports']=subprocess.check_output([str(RUNTIME/'bin/python'),'-c',
        'import torch,astate,astate._core.astate_cpp as n; print(torch.__version__,astate.__version__,n.__file__)'],
        env=astate_env,text=True)
    row['astate']={key:manifest[key] for key in ['source_revision','transport','extension_sha256','utrans_sha256']}
    row['astate']['source_patch_sha256']=manifest.get('source_patch_sha256')
(BASE/'prepared.json').write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps(row),flush=True)
