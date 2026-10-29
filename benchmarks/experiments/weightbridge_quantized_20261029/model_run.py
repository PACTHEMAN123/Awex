"""Real Qwen3-30B-A3B qualification on the exact FP8 fusion topology."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

BASE = Path('/tmp/weightbridge-quantized-1029')
SOURCE = BASE / 'source'
STAGE = BASE / 'model-stage'
IPS = ['11.18.56.89', '11.18.50.220', '11.18.50.148', '11.17.132.185']
DEPS = (f'{BASE}:{BASE}/model-verl:/tmp/verl-nccl-1020-deps:'
        '/tmp/weightbridge-memory-1029/weightbridge:/tmp/comm-memory-1021-mooncake:'
        '/tmp/figure11-1025/awex')
NATIVE_SHA = '6de1b04659d22ebe6972f26d76344714eb11b61cd75bc1b3b5b2fcb036c21d0a'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


@ray.remote(num_cpus=0)
def prepare(verl_archive, revision):
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == revision
    shutil.copytree('/tmp/fp8-fusion-1029/stage', STAGE, dirs_exist_ok=True)
    shutil.copytree(SOURCE / 'python/shardstream', STAGE / 'shardstream', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(SOURCE / 'benchmarks/shardstream_benchmarks', STAGE / 'shardstream_benchmarks', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    with tarfile.open(fileobj=io.BytesIO(verl_archive), mode='r:gz') as archive:
        archive.extractall(BASE / 'model-verl', filter='data')
    native = list((STAGE / 'shardstream').glob('*.so'))
    assert len(native) == 1 and hashlib.sha256(native[0].read_bytes()).hexdigest() == NATIVE_SHA
    return dict(ip=ray.util.get_node_ip_address(), revision=revision, native_sha256=NATIVE_SHA,
                verl_archive_sha256=hashlib.sha256(verl_archive).hexdigest(),
                model_path='/mnt/fuse/verl-e2e/models/Qwen3-30B-A3B')


@ray.remote(num_cpus=0)
def collect(run_dir):
    path = Path(run_dir)
    assert path.is_relative_to(BASE / 'model-runs')
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        if path.exists():
            for p in sorted(path.rglob('*')):
                if p.is_file():
                    archive.add(p, arcname=str(p.relative_to(path)))
    return dict(ip=ray.util.get_node_ip_address(), data=stream.getvalue(),
                gpu_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader'], text=True))


def calls(fn, arguments):
    ray.init(address=IPS[0] + ':6381', runtime_env={'env_vars': {
        'PYTHONPATH': str(SOURCE / 'benchmarks/experiments/weightbridge_quantized_20261029') + ':' + str(STAGE) + ':' + DEPS}},
        logging_level='ERROR')
    try:
        nodes = {n['NodeManagerAddress']: n['NodeID'] for n in ray.nodes() if n['Alive']}
        return ray.get([fn.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            nodes[ip], soft=False)).remote(*arguments) for ip in IPS], timeout=180)
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    args = parser.parse_args()
    root = BASE / 'model-results' / args.label
    root.mkdir(parents=True, exist_ok=False)
    lock = open('/mnt/fuse/verl-e2e/runs/five-model-baselines-20261012/matrix/active.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for p in sorted(Path('/tmp/verl-nccl-1020-verl').rglob('*')):
                if p.is_file() and not any(part in ('.git', '__pycache__') for part in p.parts):
                    archive.add(p, arcname=str(p.relative_to('/tmp/verl-nccl-1020-verl')))
        deployment = calls(prepare, (stream.getvalue(), revision))
        save(root / 'deployment.json', deployment)
        env = dict(os.environ, PYTHONPATH=str(STAGE) + ':' + DEPS,
                   SHARDSTREAM_RECIPE_REPOSITORY=str(SOURCE),
                   LD_PRELOAD='/tmp/fp8-fusion-1029/nccl/lib/libnccl.so.2',
                   LD_LIBRARY_PATH='/tmp/fp8-fusion-1029/nccl/lib:/tmp/verl-nccl-1020-deps/nvidia/cuda_runtime/lib')
        test_env = dict(env, PYTHONPATH=env['PYTHONPATH'] + ':/tmp/fp8-fusion-1029/build-deps')
        tests = [str(SOURCE / 'tests' / name) for name in ['test_weightbridge_fp8_model.py',
            'test_weightbridge_quantized.py', 'test_weightbridge_native.py', 'test_fp8_model_profile.py']]
        with (root / 'tests.log').open('w') as log:
            tested = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], env=test_env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180)
        assert tested.returncode == 0, ('focused tests failed', tested.returncode)
        command = [sys.executable, '-m', 'shardstream_benchmarks.recipe_runner', 'qwen3-30b-a3b',
            '--ray-address', IPS[0] + ':6381', '--training-hosts', *IPS[2:], '--rollout-hosts', *IPS[:2],
            '--model-path', '/mnt/fuse/verl-e2e/models/Qwen3-30B-A3B', '--source', str(SOURCE),
            '--stage', str(STAGE), '--run-root', str(BASE / 'model-runs'), '--output', str(root / 'outcome.json'),
            '--layout', 'qwen30-tp2-cp2-ep8-r8tp2', '--rollout-replicas', '4', '--gpu-placement', 'nic-spread',
            '--publication-mechanism', 'weightbridge_gather_fp8', '--publication-bucket-mb', '2048',
            '--model-weight-profile', '--weight-format', 'fp8', '--disable-prefix-caching',
            '--rollout-gpu-memory-utilization', '0.35', '--extra-python-path', DEPS,
            '--extra-library-path', '/tmp/verl-nccl-1020-deps/nvidia/cuda_runtime/lib',
            '--runtime', '/tmp/verl-nccl-1020-runtime', '--nccl-lib', '/tmp/fp8-fusion-1029/nccl/lib',
            '--num-updates', str(args.updates), '--warmup-updates', str(args.warmup), '--timeout', '3600',
            '--dp-source-policy', 'stream', '--dp-source-ring-order', 'source-node', '--replica-routing', 'logical']
        save(root / 'case.json', dict(command=command, revision=revision,
             matched_reference='/tmp/fp8-fusion-resume-1029/results', native_sha256=NATIVE_SHA))
        save(root / 'active.json', dict(state='running', started=time.time(), pid=os.getpid()))
        with (root / 'controller.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=4200)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    code = process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
        (root / 'exit_code').write_text(str(code) + '\n')
        if (root / 'outcome.json').exists():
            outcome = json.loads((root / 'outcome.json').read_text())
            manifest = []
            for row in calls(collect, (outcome['run_dir'],)):
                data = row.pop('data')
                filename = row['ip'] + '.tar.gz'
                (root / filename).write_bytes(data)
                manifest.append(dict(row, file=filename, sha256=hashlib.sha256(data).hexdigest()))
            save(root / 'manifest.json', manifest)
        assert code == 0, ('real-model recipe failed', code)
        save(root / 'active.json', dict(state='complete', finished=time.time()))
    except BaseException:
        save(root / 'active.json', dict(state='failed', finished=time.time(), traceback=traceback.format_exc()))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
