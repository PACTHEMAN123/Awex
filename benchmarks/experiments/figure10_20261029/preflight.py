"""Verify immutable existing assets against the Figure 9 checkpoint and sources."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from protocol import ASSETS, BASE, DEPS, IPS, REVISION, SOURCE, STAGE, load, save


@ray.remote(num_cpus=0)
def inspect(expected_assets, expected_wb):
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def output(*args):
        return subprocess.check_output(args, text=True).strip()

    assert not output('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader')
    assert output('git', '-C', str(SOURCE), 'rev-parse', 'HEAD') == REVISION
    assert not output('git', '-C', str(SOURCE), 'status', '--porcelain')
    for source_dir, package in [('python/shardstream', 'shardstream'),
                                ('benchmarks/shardstream_benchmarks', 'shardstream_benchmarks')]:
        for source in (SOURCE / source_dir).rglob('*.py'):
            assert sha(source) == sha(STAGE / package / source.relative_to(SOURCE / source_dir))
    for relative, digest in expected_wb.items():
        assert sha(ASSETS / 'weightbridge' / relative) == digest, relative
    missing_metadata = []
    for row in expected_assets:
        path = Path('/mnt/fuse/verl-e2e') / row['path']
        if path.name == '_hf_internal_model_info.json' and not path.exists():
            missing_metadata.append(row['path'])
            continue
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    assert sha(ASSETS / 'nccl/lib/libnccl.so.2') == '78103983bb558aef7277a0c05a9a5963e619a69f6e77ece3c449fb045a3368d4'
    inherited = json.loads((ASSETS / 'prepared.json').read_text())
    for name, digest in inherited['native_sha256'].items():
        assert sha(STAGE / 'shardstream' / name) == digest
    environment = dict(os.environ, PYTHONPATH=str(STAGE) + ':' + DEPS,
                       LD_PRELOAD=str(ASSETS / 'nccl/lib/libnccl.so.2'))
    versions = json.loads(subprocess.check_output([
        '/tmp/verl-nccl-1020-runtime/bin/python', '-c',
        'import ctypes,json,os,torch,ray,vllm,wbridge; v=ctypes.c_int(); '
        'assert ctypes.CDLL(os.environ["LD_PRELOAD"]).ncclGetVersion(ctypes.byref(v))==0; '
        'print(json.dumps(dict(torch=torch.__version__,ray=ray.__version__,vllm=vllm.__version__,nccl=v.value)))'],
        env=environment, text=True))
    assert versions == dict(torch='2.13.0+cu132', ray='2.58.0', vllm='0.27.0', nccl=23102)
    BASE.mkdir(exist_ok=True)
    row = dict(passed=True, ip=ray.util.get_node_ip_address(), hostname=output('hostname'),
               user=output('id', '-un'), cwd=os.getcwd(), gpus=output('nvidia-smi', '-L'),
               revision=REVISION, model_files_verified=len(expected_assets) - len(missing_metadata),
               absent_optional_downloader_metadata=missing_metadata,
               weightbridge_files_verified=len(expected_wb),
               inherited_environment=inherited, runtime_versions=versions,
               oss_exists=Path('/mnt/fuse/oss/xiaopac.xjy').is_dir())
    (BASE / 'prepared.json').write_text(json.dumps(row, indent=2) + '\n')
    return row


if __name__ == '__main__':
    reference = load(BASE / 'reference.json')
    ray.init(address=IPS[0] + ':6381', logging_level='ERROR')
    try:
        nodes = {node['NodeManagerAddress']: node['NodeID'] for node in ray.nodes() if node['Alive']}
        tasks = [inspect.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[ip], soft=False)).remote(
            reference['model_assets'], reference['weightbridge_source_sha256']) for ip in IPS]
        rows, failures = [], []
        for task in tasks:
            try:
                rows.append(ray.get(task, timeout=600))
            except Exception as error:
                failures.append(str(error))
        assert not failures, failures
        save(BASE / 'deployment.json', rows)
        print('All four nodes passed Figure 9 asset and runtime preflight', flush=True)
    finally:
        ray.shutdown()
