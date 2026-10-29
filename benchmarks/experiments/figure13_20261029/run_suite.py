"""Two disjoint two-host sweeps under the existing campaign-wide GPU lock."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from legacy_summary import analyze, parse_events

BASE = Path('/tmp/weightbridge-memory-1029')
ROOT = BASE / 'results'
SOURCE, STAGE = BASE / 'source', BASE / 'stage'
IPS = ['11.18.56.89', '11.18.50.220', '11.18.50.148', '11.17.132.185']
DEPS = '/tmp/verl-nccl-1020-deps:/tmp/verl-nccl-1020-verl:/tmp/weightbridge-fig9-1022/weightbridge:/tmp/comm-memory-1021-mooncake'
DURABLE = Path('/mnt/fuse/verl-e2e/runs/weightbridge-memory-20261029')
PAIRS = {'a': (IPS[:2], [128, 64, 256, 512]), 'b': (IPS[2:], [32, 16, 8, 256])}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


@ray.remote(num_cpus=0)
def inventory():
    data = json.loads((BASE / 'prepared.json').read_text())
    data['gpu_processes'] = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
    data['captured'] = time.time()
    data['shm_free'] = os.statvfs('/dev/shm').f_bavail * os.statvfs('/dev/shm').f_frsize
    return data


@ray.remote(num_cpus=0)
def collect(run_dir):
    path = Path(run_dir)
    assert path.is_relative_to(BASE / 'runs') and path.is_dir()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for p in sorted(path.rglob('*')):
            if p.is_file():
                archive.add(p, arcname=str(p.relative_to(path)))
    return dict(ip=ray.util.get_node_ip_address(), data=stream.getvalue())


def calls(task, ips, *args):
    return ray.get([task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        NODES[ip], soft=False)).remote(*args) for ip in ips], timeout=180)


def mirror(directory):
    shutil.copytree(directory, DURABLE / directory.name, dirs_exist_ok=True)


def audit(directory, case):
    result = analyze(directory)
    caps, guards = [], []
    for entry in json.loads((directory / 'manifest.json').read_text()):
        with tarfile.open(directory / entry['file']) as archive:
            for member in archive.getmembers():
                if member.name.endswith('.log'):
                    text = archive.extractfile(member).read().decode(errors='replace')
                    caps.extend(parse_events(text, 'WB_ROUND_CAP'))
                    guards.extend(parse_events(text, 'WB_CHECKPOINT_GUARD'))
    assert len(caps) == len(guards) == 16, (len(caps), len(guards))
    assert all(r['effective_bytes'] == r['requested_bytes'] == case['bucket'] * 2**20
               and r['logical_tensor_cap_bytes'] == case['logical_tensor_cap_mib'] * 2**20
               for r in caps), caps
    assert all(g['verified'] for g in guards)
    result.update(pair=case['pair'], repeat=case['repeat'],
                  logical_tensor_cap_mib=case['logical_tensor_cap_mib'],
                  round_cap_processes=len(caps), checkpoint_guards=len(guards))
    save(directory / 'summary.json', result)
    save(directory / 'complete.json', dict(passed=True, finished=time.time()))
    return result


def run_case(pair, ips, cap, repeat):
    logical_cap = min(cap, 128)
    label = f'pair-{pair}-b{cap}-l{logical_cap}-r{repeat}'
    directory = ROOT / label
    if (directory / 'complete.json').exists():
        return json.loads((directory / 'summary.json').read_text())
    directory.mkdir(exist_ok=False)
    before = calls(inventory, ips)
    save(directory / 'inventory.json', before)
    assert all(not r['gpu_processes'].strip() for r in before), before
    assert len({r['revision'] for r in before}) == 1
    cmd = [sys.executable, '-m', 'shardstream_benchmarks.recipe_runner', 'qwen3-30b-a3b',
           '--ray-address', IPS[0] + ':6381', '--training-hosts', ips[0], '--rollout-hosts', ips[1],
           '--model-path', '/mnt/fuse/verl-e2e/models/Qwen3-30B-A3B',
           '--source', str(SOURCE), '--stage', str(STAGE), '--run-root', str(BASE / 'runs'),
           '--output', str(directory / 'outcome.json'), '--layout', 'qwen30-train8-tp4-cp2-ep8-r1tp4',
           '--rollout-replicas', '2', '--publication-mechanism', 'weightbridge_native',
           '--publication-bucket-mb', str(cap), '--weightbridge-logical-tensor-cap-mb', str(logical_cap),
           '--model-weight-profile', '--communication-memory-profile', '--disable-prefix-caching',
           '--extra-python-path', DEPS, '--extra-library-path', '/tmp/verl-nccl-1020-deps/nvidia/cuda_runtime/lib',
           '--runtime', '/tmp/verl-nccl-1020-runtime', '--nccl-lib', str(BASE / 'nccl/lib'),
           '--num-updates', '9', '--warmup-updates', '2', '--timeout', '2400',
           '--dp-source-policy', 'stream', '--dp-source-ring-order', 'source-node', '--replica-routing', 'logical']
    case = dict(label=label, pair=pair, repeat=repeat, backend='weightbridge_native', bucket=cap,
                logical_tensor_cap_mib=logical_cap, channels=None, depth=None, updates=9, warmup=2,
                replicas=2, weightbridge_cuda13=True, command=cmd, revision=before[0]['revision'],
                training_hosts=[ips[0]], rollout_hosts=[ips[1]])
    save(directory / 'case.json', case)
    save(ROOT / f'active-{pair}.json', dict(state='running', label=label, started=time.time()))
    print(json.dumps(dict(event='start', label=label)), flush=True)
    env = dict(os.environ, PYTHONPATH=str(STAGE) + ':' + DEPS, SHARDSTREAM_RECIPE_REPOSITORY=str(SOURCE))
    with (directory / 'controller.log').open('w') as log:
        child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        save(directory / 'process.json', dict(pid=child.pid, started=time.time()))
        try:
            code = child.wait(timeout=3000)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                code = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
    save(directory / 'exit.json', dict(code=code))
    if (directory / 'outcome.json').exists():
        outcome = json.loads((directory / 'outcome.json').read_text())
        manifest = []
        for row in calls(collect, ips, outcome['run_dir']):
            name = row['ip'] + '.tar.gz'
            (directory / name).write_bytes(row['data'])
            manifest.append(dict(file=name, sha256=hashlib.sha256(row['data']).hexdigest(), bytes=len(row['data'])))
        save(directory / 'manifest.json', manifest)
    after = calls(inventory, ips)
    for _ in range(12):
        if all(not r['gpu_processes'].strip() for r in after):
            break
        time.sleep(5)
        after = calls(inventory, ips)
    save(directory / 'final-inventory.json', after)
    mirror(directory)
    assert all(not r['gpu_processes'].strip() for r in after), 'Pair GPUs not released; stop this lane'
    assert code == 0, (label, code)
    result = audit(directory, case)
    mirror(directory)
    print(json.dumps(dict(event='passed', label=label, p50_ms=result['p50_ms'],
                         resident_gib=result['steady_rank_medians_by_role']['rollout']['device_resident_delta_gib']['max'])), flush=True)
    return result


def run_pair(pair):
    ips, caps = PAIRS[pair]
    rows = []
    try:
        for repeat in range(3):
            for cap in caps if repeat % 2 == 0 else list(reversed(caps)):
                rows.append(run_case(pair, ips, cap, repeat))
                save(ROOT / f'results-{pair}.json', rows)
        save(ROOT / f'active-{pair}.json', dict(state='complete', launches=len(rows), finished=time.time()))
        return True
    except BaseException:
        save(ROOT / f'active-{pair}.json', dict(state='failed', traceback=traceback.format_exc(), finished=time.time()))
        traceback.print_exc()
        return False


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    DURABLE.mkdir(parents=True, exist_ok=True)
    with open('/mnt/fuse/verl-e2e/runs/five-model-baselines-20261012/matrix/active.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ray.init(address=IPS[0] + ':6381', logging_level='ERROR')
        try:
            NODES = {n['NodeManagerAddress']: n['NodeID'] for n in ray.nodes() if n['Alive']}
            assert set(IPS) <= set(NODES)
            before = calls(inventory, IPS)
            assert all(not r['gpu_processes'].strip() for r in before), before
            save(ROOT / 'initial-inventory.json', before)
            save(ROOT / 'active.json', dict(state='running', pid=os.getpid(), started=time.time(), planned_launches=24))
            with ThreadPoolExecutor(max_workers=2) as pool:
                ok = list(pool.map(run_pair, PAIRS))
            save(ROOT / 'active.json', dict(state='complete' if all(ok) else 'needs-investigation', finished=time.time()))
            shutil.copytree(ROOT, DURABLE, dirs_exist_ok=True)
        finally:
            ray.shutdown()
