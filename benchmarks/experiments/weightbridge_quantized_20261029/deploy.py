"""Install a committed incremental bundle without modifying prior campaigns."""
import hashlib
import json
from pathlib import Path
import subprocess

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

BASE = Path('/tmp/weightbridge-quantized-1029')
IPS = ['11.18.56.89', '11.18.50.220', '11.18.50.148', '11.17.132.185']


@ray.remote(num_cpus=0)
def prepare(bundle, revision):
    BASE.mkdir(exist_ok=True)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    (BASE / 'source.bundle').write_bytes(bundle)
    source = BASE / 'source'
    if not source.exists():
        subprocess.run(['git', 'clone', '--shared', '/tmp/weightbridge-memory-1029/source', str(source)], check=True)
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    subprocess.run(['git', '-C', str(source), 'fetch', str(BASE / 'source.bundle'), 'HEAD'], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == revision
    kernel = Path('/tmp/verl-nccl-1020-verl/verl/utils/kernel/fp8_kernel.py')
    return dict(ip=ray.util.get_node_ip_address(), revision=revision,
                quantizer_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
                source_bundle_sha256=hashlib.sha256(bundle).hexdigest(),
                gpus=subprocess.check_output(['nvidia-smi', '-L'], text=True))


if __name__ == '__main__':
    import sys
    ray.init(address=IPS[0] + ':6381', logging_level='ERROR')
    try:
        nodes = {n['NodeManagerAddress']: n['NodeID'] for n in ray.nodes() if n['Alive']}
        bundle = (BASE / 'source.bundle').read_bytes()
        rows = ray.get([prepare.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            nodes[ip], soft=False)).remote(bundle, sys.argv[1]) for ip in IPS], timeout=180)
        (BASE / 'deployment.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(rows), flush=True)
    finally:
        ray.shutdown()
