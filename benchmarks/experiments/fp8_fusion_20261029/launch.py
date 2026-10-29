"""Deploy only controller helpers; preserve the pinned measured build."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

BASE = Path('/tmp/fp8-fusion-resume-1029')
IPS = ['11.18.56.89', '11.18.50.220', '11.18.50.148', '11.17.132.185']


@ray.remote(num_cpus=0)
def prepare(evidence):
    assets = Path('/tmp/fp8-fusion-1029')
    assert not subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert subprocess.check_output(['git', '-C', str(assets / 'source'), 'rev-parse', 'HEAD'],
                                   text=True).strip() == '67c9cfeebc5f593fbd0e8f63001c2d64e766872a'
    for source_dir, package in [('python/shardstream', 'shardstream'),
                                ('benchmarks/shardstream_benchmarks', 'shardstream_benchmarks')]:
        source = assets / 'source' / source_dir
        for path in source.rglob('*.py'):
            assert path.read_bytes() == (assets / 'stage' / package / path.relative_to(source)).read_bytes(), str(path)
    BASE.mkdir(exist_ok=True)
    (BASE / 'evidence.py').write_bytes(evidence)
    return dict(ip=ray.util.get_node_ip_address(), passed=True,
                helper_sha256=hashlib.sha256(evidence).hexdigest(),
                hostname=subprocess.check_output(['hostname'], text=True).strip())


if __name__ == '__main__':
    ray.init(address=IPS[0] + ':6381', logging_level='ERROR')
    try:
        nodes = {n['NodeManagerAddress']: n['NodeID'] for n in ray.nodes() if n['Alive']}
        evidence = (BASE / 'evidence.py').read_bytes()
        rows = [ray.get(prepare.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            nodes[ip], soft=False)).remote(evidence), timeout=120) for ip in IPS]
        (BASE / 'deployment.json').write_text(json.dumps(rows, indent=2) + '\n')
    finally:
        ray.shutdown()
    environment = dict(os.environ, PYTHONPATH=str(BASE))
    with (BASE / 'suite.log').open('x') as log:
        child = subprocess.Popen(['/tmp/verl-nccl-1020-runtime/bin/python', str(BASE / 'run_suite.py')],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=environment, cwd=BASE, start_new_session=True)
    record = dict(pid=child.pid, started=time.time(), controller_sha256=hashlib.sha256(
        (BASE / 'run_suite.py').read_bytes()).hexdigest())
    (BASE / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
