"""Four-node, frameworkless qualification of the experimental WB FP8 frontend.

Two trainer ranks on two hosts; two independent full-tensor receivers on the
other two hosts. This does not load a model or measure generation performance.
"""

import argparse
import asyncio
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

BASE = Path('/tmp/weightbridge-quantized-1029')
IPS = ['11.18.56.89', '11.18.50.220', '11.18.50.148', '11.17.132.185']
WB = Path('/tmp/weightbridge-memory-1029/weightbridge')
WB_REVISION = '17ea1bc6370424d52152880834a94659db5002ec'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def values(size, index, epoch, device, partition=None):
    """Changing deterministic values with exact power-of-two block scales."""
    import torch
    rows = torch.arange(size, device=device, dtype=torch.int32)[:, None]
    cols = torch.arange(size, device=device, dtype=torch.int32)[None, :]
    if partition is not None:
        sl = partition.slices()
        rows, cols = rows[sl[0]], cols[:, sl[1]]
    v = ((rows * 17 + cols * 29 + index * 13 + epoch * 7) % 801 - 400).float()
    factor = torch.pow(2.0, ((rows // 128 + cols // 128 + epoch) % 5 - 2).float()) / 16
    v = torch.where((rows % 128 == 0) & (cols % 128 == 0), 448.0, v)
    return (v * factor).to(torch.bfloat16)


def reference_payload(size, index, epoch, device):
    import torch
    x = values(size, index, epoch, device)
    blocks = x.float().view(size // 128, 128, size // 128, 128)
    scales = blocks.abs().amax((1, 3)) / 448.0
    q = (blocks / scales[:, None, :, None]).reshape(size, size).to(torch.float8_e4m3fn)
    return q.view(torch.uint8), scales


@ray.remote(num_cpus=0)
class Barrier:
    def __init__(self):
        self.arrivals = {}
        self.events = {}

    async def wait(self, epoch, participant):
        arrivals = self.arrivals.setdefault(epoch, set())
        if participant in arrivals:
            raise RuntimeError('Duplicate barrier arrival')
        arrivals.add(participant)
        event = self.events.setdefault(epoch, asyncio.Event())
        if len(arrivals) == 4:
            event.set()
        await asyncio.wait_for(event.wait(), 300)


@ray.remote(num_cpus=1, num_gpus=1)
class Worker:
    def initialize(self, role, rank, size, mode, ports):
        import torch
        import torch.distributed as dist
        from shardstream.integrations.publication.weightbridge_quantized import (
            GatherBeforeQuantize, MatrixPartition,
        )
        from wbridge.frontend.adapters import AdapterContext, BaseAdapter, ReceiverAdapter, SenderAdapter
        from wbridge.backend.sender import SenderArgs
        torch.cuda.set_device(0)
        self.role, self.rank, self.size, self.mode = role, rank, size, mode
        self.device = torch.device('cuda:0')
        self.names = ['matrix.row', 'matrix.column']
        self.parts = {name: MatrixPartition((size, size), index, rank, 2)
                      for index, name in enumerate(self.names)} if role == 'sender' else {}
        self.source = {name: values(size, i, 0, self.device, self.parts.get(name))
                       for i, name in enumerate(self.names)}

        def context(payload, quantized=False):
            shapes = {name: ((size // 128, size // 128) if name.endswith('.scale') else (size, size))
                      for name in payload}
            factories = {}
            for name in payload:
                base = name.removesuffix('.scale')
                i = self.names.index(base)
                if quantized:
                    factories[name] = lambda i=i, scale=name.endswith('.scale'): reference_payload(
                        size, i, 0, self.device)[int(scale)]
                else:
                    factories[name] = lambda i=i: values(size, i, 0, self.device)

            def loader(fetcher):
                for name in payload:
                    value = fetcher[name]()
                    base = name.removesuffix('.scale')
                    if role == 'sender':
                        value = value[self.parts[base].slices(scales=name.endswith('.scale'))]
                    payload[name].copy_(value)
            return AdapterContext(factories, shapes, lambda: payload, loader, rank if role == 'sender' else 0)

        if role == 'sender':
            # Exercise the unmodified extractor on the original BF16 loader too.
            original = BaseAdapter(context(self.source))
            for i, name in enumerate(self.names):
                assert torch.equal(self.source[name], values(size, i, 0, self.device, self.parts[name]))
            self.original_spec_hash = hashlib.sha256(repr(original.load_spec.entries).encode()).hexdigest()
            dist.init_process_group('nccl', init_method=f'tcp://{IPS[2]}:{ports[2]}',
                                    rank=rank, world_size=2, timeout=timedelta(seconds=180),
                                    device_id=self.device)
            if mode != 'bf16':
                from verl_fp8_kernel import scaled_fp8_blockwise_triton
                self.frontend = GatherBeforeQuantize(self.source, self.parts, scaled_fp8_blockwise_triton)
                self.frontend.prepare()
                self.payload = self.frontend.payload
            else:
                self.payload = self.source
            args = SenderArgs(2, [f'tcp://{ip}:{port}' for ip, port in zip(IPS[:2], ports[:2])],
                              IPS[2], ports[3], protocol='rdma')
            self.adapter = SenderAdapter(context(self.payload, mode != 'bf16'), args)
        else:
            from wbridge.backend import coordinator
            from wbridge.backend.control_channel import coordinator_ipc
            self.ipc = coordinator_ipc(ports[rank])
            self.coordinator = coordinator.spawn(self.ipc, ports[rank])
            if mode != 'bf16':
                self.payload = {}
                for i, name in enumerate(self.names):
                    q, s = reference_payload(size, i, 0, self.device)
                    self.payload[name], self.payload[name + '.scale'] = q, s
            else:
                self.payload = self.source
            self.adapter = ReceiverAdapter(context(self.payload, mode != 'bf16'), self.ipc, 1)
            # Start with poison so an omitted first transfer cannot pass.
            for tensor in self.payload.values():
                tensor.fill_(0)
        self.pointers = {name: t.data_ptr() for name, t in self.payload.items()}
        return dict(role=role, rank=rank, pid=os.getpid(), host=ray.util.get_node_ip_address(),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    payload_bytes=sum(t.numel() * t.element_size() for t in self.payload.values()),
                    original_spec_hash=getattr(self, 'original_spec_hash', None),
                    payload_spec_hash=hashlib.sha256(repr(self.adapter.load_spec.entries).encode()).hexdigest())

    def run(self, count, barrier):
        import torch
        import torch.distributed as dist
        rows = []
        if self.role == 'sender':
            self.adapter.connect()
            sender = self.adapter.sender
            router = sender.router
            native_wire_bytes = sum(spec.nbytes(sender.dtype_spec)
                                    for _, overlaps in router.local_rounds for spec in overlaps.values())
            plan = dict(rounds=len(router.global_rounds), native_wire_bytes=native_wire_bytes,
                        plan_hash=hashlib.sha256(repr([(fs.entries, {r: sp.entries for r, sp in ov.items()})
                            for fs, ov in router.local_rounds]).encode()).hexdigest())
        else:
            plan = None
        for epoch in range(count):
            if self.role == 'sender':
                for i, name in enumerate(self.names):
                    self.source[name].copy_(values(self.size, i, epoch + 1, self.device, self.parts[name]))
                outside = self.frontend.prepare() if self.mode == 'prepared_fp8' else {}
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base_memory = torch.cuda.memory_allocated()
                start = time.perf_counter()
                if self.mode == 'gather_fp8':
                    row = self.frontend.publish(self.adapter)
                else:
                    event = self.adapter.send_weights()
                    if event is not None:
                        event.synchronize()
                    self.adapter.wait_send_complete()
                    row = dict(complete_ms=(time.perf_counter() - start) * 1000)
                row.update(epoch=epoch, allocated_before_bytes=base_memory,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           prepare_outside_timing=outside)
                self.adapter.flush_profile_outputs()
            else:
                deadline = time.monotonic() + 240
                while not self.adapter.poll_requests():
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'Receiver timed out at epoch {epoch}')
                    time.sleep(0.0005)
                torch.cuda.synchronize()
                # Independent full-value oracle, excluded from transfer timings.
                checked = 0
                for i, name in enumerate(self.names):
                    if self.mode == 'bf16':
                        expected = {name: values(self.size, i, epoch + 1, self.device)}
                    else:
                        q, s = reference_payload(self.size, i, epoch + 1, self.device)
                        expected = {name: q, name + '.scale': s}
                    for key, value in expected.items():
                        assert torch.equal(self.payload[key], value), (epoch, key)
                        assert self.payload[key].data_ptr() == self.pointers[key]
                        checked += value.numel()
                row = dict(epoch=epoch, verified=True, checked_elements=checked)
                self.adapter.flush_profile_outputs()
            rows.append(row)
            ray.get(barrier.wait.remote(epoch, f'{self.role}:{self.rank}'), timeout=300)
        return dict(role=self.role, rank=self.rank, rows=rows, plan=plan)

    def stop(self):
        if hasattr(self, 'coordinator'):
            self.coordinator.terminate()
            self.coordinator.wait(timeout=10)


@ray.remote(num_cpus=0)
def inventory():
    prepared = json.loads((WB.parent / 'prepared.json').read_text())
    revision = prepared['weightbridge_revision']
    assert revision == WB_REVISION
    for path, digest in prepared['weightbridge_python_sha256'].items():
        assert hashlib.sha256((WB / path).read_bytes()).hexdigest() == digest, path
    return dict(ip=ray.util.get_node_ip_address(), hostname=subprocess.check_output(['hostname'], text=True).strip(),
                gpu_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True),
                wb_revision=revision,
                wb_python_sha256=prepared['weightbridge_python_sha256'],
                revision=subprocess.check_output(['git', '-C', str(BASE / 'source'), 'rev-parse', 'HEAD'], text=True).strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--modes', nargs='+', default=['bf16', 'gather_fp8', 'prepared_fp8'],
                        choices=['bf16', 'gather_fp8', 'prepared_fp8'])
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    root = BASE / 'results' / args.label
    root.mkdir(parents=True, exist_ok=False)
    lock = open('/mnt/fuse/verl-e2e/runs/five-model-baselines-20261012/matrix/active.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(PYTHONPATH=str(BASE) + ':' + os.environ['PYTHONPATH'], LD_LIBRARY_PATH=os.environ['LD_LIBRARY_PATH'],
               LD_PRELOAD=os.environ['LD_PRELOAD'], NCCL_SOCKET_IFNAME='eth0', GLOO_SOCKET_IFNAME='eth0',
               NCCL_IB_DISABLE='0', NCCL_IB_GID_INDEX='3', NCCL_DEBUG='INFO', NCCL_DEBUG_SUBSYS='INIT,NET,ENV',
               NCCL_CUMEM_ENABLE='1', TORCHELASTIC_USE_AGENT_STORE='False',
               WBRIDGE_SPECGEN_PROBE_DEVICE='cuda', WBRIDGE_ROUND_CAP_BYTES=str(2 << 30),
               WBRIDGE_LOGICAL_TENSOR_CAP_BYTES=str(128 << 20), OMP_NUM_THREADS='4')
    ray.init(address=IPS[0] + ':6381', runtime_env={'env_vars': env}, logging_level='ERROR')
    nodes = {n['NodeManagerAddress']: n['NodeID'] for n in ray.nodes() if n['Alive']}
    sched = [NodeAffinitySchedulingStrategy(nodes[ip], soft=False) for ip in IPS]

    def inspect():
        return ray.get([inventory.options(scheduling_strategy=s).remote() for s in sched], timeout=90)

    all_results = []
    try:
        before = inspect()
        save(root / 'inventory.json', before)
        assert all(not r['gpu_processes'].strip() for r in before), before
        for repeat in range(args.repeat):
            modes = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
            for mode in modes:
                label = f'{mode}-r{repeat}'
                save(root / 'active.json', dict(state='running', case=label, time=time.time()))
                print(json.dumps(dict(event='start', case=label)), flush=True)
                workers = [Worker.options(scheduling_strategy=s).remote() for s in sched]
                barrier = Barrier.remote()
                ports = [62710, 62711, 62712, 62713]
                try:
                    initialized = ray.get([w.initialize.remote('receiver' if i < 2 else 'sender',
                        i % 2, args.size, mode, ports) for i, w in enumerate(workers)], timeout=360)
                    save(root / (label + '-initialized.json'), initialized)
                    results = ray.get([w.run.remote(args.updates, barrier) for w in workers], timeout=600)
                    save(root / (label + '-raw.json'), results)
                    assert all(row['verified'] for r in results if r['role'] == 'receiver' for row in r['rows'])
                    senders = [r for r in results if r['role'] == 'sender']
                    measured = [max(r['rows'][i]['complete_ms'] for r in senders)
                                for i in range(args.warmup, args.updates)]
                    summary = dict(case=label, mode=mode, repeat=repeat, size=args.size,
                        p50_ms=statistics.median(measured), min_ms=min(measured), max_ms=max(measured),
                        timed_updates=len(measured), checked_receiver_updates=2 * args.updates,
                        original_bf16_bytes=4 * args.size * args.size,
                        native_wire_bytes=sum(r['plan']['native_wire_bytes'] for r in senders),
                        logical_gather_egress_bytes=(4 * args.size * args.size if mode != 'bf16' else 0),
                        excluded_preparation=(mode == 'prepared_fp8'))
                    all_results.append(summary)
                    save(root / 'summary.json', all_results)
                    print(json.dumps(summary), flush=True)
                finally:
                    for w in workers:
                        try:
                            ray.get(w.stop.remote(), timeout=15)
                        except Exception:
                            pass
                        ray.kill(w)
                    ray.kill(barrier)
                for _ in range(15):
                    after = inspect()
                    if all(not r['gpu_processes'].strip() for r in after):
                        break
                    time.sleep(2)
                save(root / (label + '-final-inventory.json'), after)
                assert all(not r['gpu_processes'].strip() for r in after), after
        save(root / 'active.json', dict(state='complete', time=time.time()))
    except BaseException:
        import traceback
        save(root / 'active.json', dict(state='failed', traceback=traceback.format_exc()))
        raise
    finally:
        ray.shutdown()
        lock.close()


if __name__ == '__main__':
    main()
