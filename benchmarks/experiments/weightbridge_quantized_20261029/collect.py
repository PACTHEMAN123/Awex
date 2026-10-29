"""Collect only worker logs belonging to this experiment, plus idle evidence."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

BASE = Path('/tmp/weightbridge-quantized-1029')


@ray.remote(num_cpus=0)
def collect(pids):
    root = Path('/tmp/ray/session_latest/logs')
    paths = [path for pid in pids for suffix in ['out', 'err']
             for path in root.glob(f'worker-*-{pid}.{suffix}')]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for path in sorted(set(paths)):
            archive.add(path, arcname=path.name)
    data = stream.getvalue()
    return dict(ip=ray.util.get_node_ip_address(), pids=sorted(pids), captured=time.time(),
                files=[p.name for p in paths], archive_sha256=hashlib.sha256(data).hexdigest(),
                gpu_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader'], text=True), data=data)


if __name__ == '__main__':
    pids = {}
    for path in (BASE / 'results').glob('*/*-initialized.json'):
        for row in json.loads(path.read_text()):
            pids.setdefault(row['host'], set()).add(row['pid'])
    ray.init(address='11.18.56.89:6381', logging_level='ERROR')
    try:
        nodes = {n['NodeManagerAddress']: n['NodeID'] for n in ray.nodes() if n['Alive']}
        rows = ray.get([collect.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            nodes[ip], soft=False)).remote(pids[ip]) for ip in sorted(pids)], timeout=120)
        (BASE / 'node-logs').mkdir(exist_ok=True)
        for row in rows:
            (BASE / 'node-logs' / (row['ip'] + '.tar.gz')).write_bytes(row.pop('data'))
            assert not row['gpu_processes'].strip(), row
            assert len(row['files']) == 2 * len(row['pids']), row
        (BASE / 'final-inventory-and-log-manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps([{k: v for k, v in r.items() if k != 'files'} for r in rows]))
    finally:
        ray.shutdown()
