"""Run six exclusive four-node launches and preserve every attempted result."""
import fcntl
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from protocol import ASSETS, BASE, DEPS, IPS, LAYOUTS, ORDER, REVISION, SOURCE, STAGE, WB_REVISION, audit, load, save

ROOT = BASE / 'results'


@ray.remote(num_cpus=0)
def inventory():
    return dict(ip=ray.util.get_node_ip_address(), revision=subprocess.check_output(
        ['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
        gpu_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                             '--format=csv,noheader'], text=True))


@ray.remote(num_cpus=0)
def collect(run_dir):
    path = Path(run_dir)
    assert path.is_relative_to(BASE / 'runs') and path.is_dir()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for item in sorted(path.rglob('*')):
            if item.is_file():
                archive.add(item, arcname=str(item.relative_to(path)))
    return dict(ip=ray.util.get_node_ip_address(), data=stream.getvalue())


def calls(task, *args):
    return ray.get([task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        NODES[ip], soft=False)).remote(*args) for ip in IPS], timeout=180)


def run_case(config, repeat):
    label = f'{config}-r{repeat}'
    existing = list(ROOT.glob(label + '-attempt*/complete.json'))
    if existing:
        assert len(existing) == 1
        result = audit(existing[0].parent)
        assert result == load(existing[0].parent / 'summary.json')
        return result
    attempt = 1 + len(list(ROOT.glob(label + '-attempt*')))
    directory = ROOT / f'{label}-attempt{attempt}'
    directory.mkdir()
    before = calls(inventory)
    save(directory / 'inventory.json', before)
    assert all(not row['gpu_processes'].strip() and row['revision'] == REVISION for row in before)
    command = [sys.executable, '-m', 'shardstream_benchmarks.recipe_runner', 'qwen3-30b-a3b',
               '--ray-address', IPS[0] + ':6381', '--training-hosts', *IPS[:2], '--rollout-hosts', *IPS[2:],
               '--model-path', '/mnt/fuse/verl-e2e/models/Qwen3-30B-A3B',
               '--source', str(SOURCE), '--stage', str(STAGE), '--run-root', str(BASE / 'runs'),
               '--output', str(directory / 'outcome.json'), '--layout', LAYOUTS[config][0],
               '--rollout-replicas', '4', '--publication-mechanism', 'weightbridge_native',
               '--publication-bucket-mb', '2048', '--weightbridge-logical-tensor-cap-mb', '128',
               '--model-weight-profile', '--disable-prefix-caching', '--extra-python-path', DEPS,
               '--extra-library-path', '/tmp/verl-nccl-1020-deps/nvidia/cuda_runtime/lib',
               '--runtime', '/tmp/verl-nccl-1020-runtime', '--nccl-lib', str(ASSETS / 'nccl/lib'),
               '--num-updates', '35', '--warmup-updates', '5', '--timeout', '2400']
    case = dict(config=config, repeat=repeat, attempt=attempt, label=label, layout=LAYOUTS[config][0],
                backend='weightbridge_native', updates=35, warmup=5, revision=REVISION,
                weightbridge_revision=WB_REVISION, round_cap_bytes=2**31,
                logical_tensor_cap_bytes=128 * 2**20, command=command)
    save(directory / 'case.json', case)
    save(ROOT / 'active.json', dict(state='running', directory=directory.name, started=time.time()))
    print(f'START {directory.name}', flush=True)
    environment = dict(os.environ, PYTHONPATH=str(STAGE) + ':' + DEPS,
                       SHARDSTREAM_RECIPE_REPOSITORY=str(SOURCE))
    with (directory / 'controller.log').open('w') as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        save(directory / 'process.json', dict(pid=child.pid, started=time.time()))
        try:
            code = child.wait(timeout=2700)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                code = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
    save(directory / 'exit.json', dict(code=code))
    if (directory / 'outcome.json').exists():
        outcome = load(directory / 'outcome.json')
        manifest = []
        for row in calls(collect, outcome['run_dir']):
            name = row['ip'] + '.tar.gz'
            (directory / name).write_bytes(row['data'])
            manifest.append(dict(file=name, bytes=len(row['data']), sha256=hashlib.sha256(row['data']).hexdigest()))
        save(directory / 'manifest.json', manifest)
    after = calls(inventory)
    for _ in range(12):
        if all(not row['gpu_processes'].strip() for row in after):
            break
        time.sleep(5)
        after = calls(inventory)
    save(directory / 'final-inventory.json', after)
    assert all(not row['gpu_processes'].strip() for row in after), 'Run did not release all GPUs'
    assert code == 0, (directory.name, code)
    result = audit(directory)
    save(directory / 'summary.json', result)
    save(directory / 'complete.json', dict(passed=True, finished=time.time()))
    print(f'PASS {directory.name} p50_ms={result["p50_ms"]}', flush=True)
    return result


if __name__ == '__main__':
    ROOT.mkdir(exist_ok=True)
    with open('/mnt/fuse/verl-e2e/runs/five-model-baselines-20261012/matrix/active.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ray.init(address=IPS[0] + ':6381', logging_level='ERROR')
        try:
            NODES = {node['NodeManagerAddress']: node['NodeID'] for node in ray.nodes() if node['Alive']}
            assert set(IPS) <= set(NODES)
            deployment = load(BASE / 'deployment.json')
            assert len(deployment) == 4 and all(row['passed'] for row in deployment)
            results = []
            for config, repeat in ORDER:
                if (BASE / 'STOP_AFTER_CASE').exists():
                    save(ROOT / 'active.json', dict(state='stopped', finished=time.time()))
                    break
                results.append(run_case(config, repeat))
                save(ROOT / 'progress.json', results)
            else:
                save(ROOT / 'active.json', dict(state='complete', launches=6, finished=time.time()))
        except BaseException:
            save(ROOT / 'active.json', dict(state='needs-investigation', traceback=traceback.format_exc(), finished=time.time()))
            raise
        finally:
            ray.shutdown()
