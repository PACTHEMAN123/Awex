"""Prepare isolated, version-pinned copies on the four selected Hole nodes."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

BASE = Path('/tmp/weightbridge-memory-1029')
IPS = ['11.18.56.89', '11.18.50.220', '11.18.50.148', '11.17.132.185']
REV = 'c66e8f5'
WB_REV = '17ea1bc6370424d52152880834a94659db5002ec'


def output(*cmd):
    return subprocess.check_output(cmd, text=True).strip()


@ray.remote(num_cpus=0)
def prepare(bundle, stage_archive, wb_archive):
    assert not output('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader')
    BASE.mkdir(exist_ok=True)
    source = BASE / 'source'
    if source.exists() and subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        source.rename(BASE / f'incomplete-source-{time.time_ns()}')
    if not source.exists():
        (BASE / 'source.bundle').write_bytes(bundle)
        subprocess.run(['git', 'clone', '-q', '-b', 'codex/weightbridge-small-memory-20261029',
                        str(BASE / 'source.bundle'), str(source)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', REV], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    revision = output('git', '-C', str(source), 'rev-parse', 'HEAD')
    assert revision.startswith(REV)
    assert not output('git', '-C', str(source), 'status', '--porcelain')
    stage = BASE / 'stage'
    with tarfile.open(fileobj=io.BytesIO(stage_archive)) as archive:
        archive.extractall(stage, filter='data')
    for relative, package in [('python/shardstream', 'shardstream'),
                              ('benchmarks/shardstream_benchmarks', 'shardstream_benchmarks')]:
        shutil.copytree(source / relative, stage / package, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    wb = BASE / 'weightbridge'
    with tarfile.open(fileobj=io.BytesIO(wb_archive)) as archive:
        archive.extractall(wb, filter='data')
    nccl = BASE / 'nccl/lib'
    nccl.mkdir(parents=True, exist_ok=True)
    target = nccl / 'libnccl.so.2'
    temporary = nccl / 'libnccl.so.2.partial'
    shutil.copy2('/tmp/fp8-fusion-1029/nccl/lib/libnccl.so.2', temporary)
    temporary.replace(target)
    env = dict(os.environ, LD_PRELOAD=str(target), PYTHONPATH=str(stage) +
               ':/tmp/verl-nccl-1020-deps:' + str(wb))
    versions = json.loads(output('/usr/bin/env', *[f'{k}={v}' for k, v in
        {'LD_PRELOAD': env['LD_PRELOAD'], 'PYTHONPATH': env['PYTHONPATH']}.items()],
        '/tmp/verl-nccl-1020-runtime/bin/python', '-c',
        'import ctypes,json,os,torch,ray,wbridge; v=ctypes.c_int(); '
        'rc=ctypes.CDLL(os.environ["LD_PRELOAD"]).ncclGetVersion(ctypes.byref(v)); '
        'assert rc==0; print(json.dumps(dict(torch=torch.__version__,nccl=v.value,'
        'torch_build_nccl=torch.cuda.nccl.version(),ray=ray.__version__)))'))
    assert versions['nccl'] == 23102, versions
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    row = dict(ip=ray.util.get_node_ip_address(), hostname=output('hostname'),
               cwd=os.getcwd(), user=output('id', '-un'), gpus=output('nvidia-smi', '-L'),
               revision=revision, weightbridge_revision=WB_REV, versions=versions,
               native_sha256={p.name: sha(p) for p in (stage / 'shardstream').glob('_native*.so')},
               nccl_sha256=sha(target),
               weightbridge_python_sha256={str(p.relative_to(wb)): sha(p)
                   for p in sorted((wb / 'wbridge').rglob('*.py'))},
               model_config_sha256=sha(Path('/mnt/fuse/verl-e2e/models/Qwen3-30B-A3B/config.json')),
               oss_exists=Path('/mnt/fuse/oss/xiaopac.xjy').is_dir(),
               durable_root='/mnt/fuse/verl-e2e/runs/weightbridge-memory-20261029')
    assert row['oss_exists'] and row['native_sha256']
    (BASE / 'prepared.json').write_text(json.dumps(row, indent=2) + '\n')
    return row


if __name__ == '__main__':
    ray.init(address=IPS[0] + ':6381', logging_level='ERROR')
    try:
        nodes = {n['NodeManagerAddress']: n['NodeID'] for n in ray.nodes() if n['Alive']}
        assert set(IPS) <= set(nodes), nodes
        bundle = (BASE / 'source.bundle').read_bytes()
        wb_source = '/tmp/weightbridge-fig9-1022/weightbridge'
        assert output('git', '-C', wb_source, 'rev-parse', 'HEAD') == WB_REV
        assert not output('git', '-C', wb_source, 'status', '--porcelain')
        wb_archive = subprocess.check_output(['git', '-C', wb_source, 'archive', 'HEAD'])
        stream = io.BytesIO()
        stage_source = Path('/tmp/weightbridge-fig9-1022/stage')
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for path in sorted(stage_source.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.add(path, arcname=str(path.relative_to(stage_source)))
        stage_archive = stream.getvalue()
        futures = [prepare.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            nodes[ip], soft=False)).remote(bundle, stage_archive, wb_archive) for ip in IPS]
        rows, errors = [], []
        for future in futures:
            try:
                rows.append(ray.get(future))
            except Exception as error:
                errors.append(str(error))
        assert not errors, errors
        for key in ['revision', 'native_sha256', 'nccl_sha256', 'model_config_sha256',
                    'weightbridge_python_sha256', 'versions']:
            assert len({json.dumps(r[key], sort_keys=True) for r in rows}) == 1, key
        (BASE / 'deployment.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(rows, indent=2))
    finally:
        ray.shutdown()
