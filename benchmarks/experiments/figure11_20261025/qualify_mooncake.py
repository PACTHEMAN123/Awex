"""Verify the persistent-registration Mooncake fix over three real RDMA hosts."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

BASE = Path("/tmp/figure11-1025")
DIRECTORY = BASE / "source/benchmarks/experiments/figure11_20261025"
HOSTS = [("11.18.50.148", "send", 2), ("11.18.50.220", "receive", 0), ("11.17.132.185", "receive", 1)]


@ray.remote(num_cpus=1, num_gpus=8)
def run_probe(role, rank):
    import time

    def network():
        result = {}
        for path in Path("/sys/class/infiniband").glob("*/ports/1/counters"):
            try:
                result[path.parents[2].name] = {
                    key: int((path / key).read_text())
                    for key in ("port_xmit_data", "port_rcv_data")}
            except OSError:
                continue
        return result

    def gpu_processes():
        return subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                        "--format=csv,noheader"], text=True).strip()

    ip = ray.util.get_node_ip_address()
    prepared = json.loads((BASE / "prepared.json").read_text())
    module = BASE / "awex/awex/transfer/mooncake_comm.py"
    assert hashlib.sha256(module.read_bytes()).hexdigest() == prepared["awex_transport_sha256"]
    assert prepared['registration_policy'] == 'persistent_snapshots'
    assert not gpu_processes(), "node already has active GPU processes"
    env = dict(os.environ,
               PYTHONPATH=str(BASE / "stage")+":"+str(BASE / "awex")+":/tmp/comm-memory-1021-mooncake:/tmp/verl-nccl-1020-deps",
               LD_LIBRARY_PATH="/tmp/verl-nccl-1020-nccl/nvidia/nccl/lib:/usr/local/cuda/lib64:/tmp/verl-nccl-1020-deps/nvidia/cuda_runtime/lib:/tmp/verl-nccl-1020-runtime/lib:/tmp/verl-nccl-1020-runtime/lib/python3.12/site-packages/torch/lib")
    before = network()
    command = ["/tmp/verl-nccl-1020-runtime/bin/python", "-u", str(DIRECTORY / "mooncake_probe.py"),
               role, "--local-ip", ip, "--sender-ip", HOSTS[0][0], "--rank", str(rank)]
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=15)
    after = network()
    log = BASE / f"mooncake-probe-{role}-{rank}.log"
    log.write_bytes(output)
    for _ in range(30):
        active = gpu_processes()
        if not active:
            break
        time.sleep(1)
    text = output.decode(errors="replace")
    events = [json.loads(line) for line in text.splitlines() if line.startswith('{"event": "MOONCAKE_PROBE"')]
    return dict(ip=ip, role=role, code=process.returncode, timed_out=timed_out,
                command=command, log=str(log), log_sha256=hashlib.sha256(output).hexdigest(),
                events=events, before=before, after=after, gpu_processes_after=active,
                artifact=prepared, closed=any(line.startswith('{"event": "MOONCAKE_PROBE_CLOSED"') for line in text.splitlines()), output_tail=text[-5000:])


def main():
    with open("/mnt/fuse/verl-e2e/runs/five-model-baselines-20261012/matrix/active.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ray.init(address="11.18.56.89:6381", logging_level="ERROR")
        try:
            nodes = {row["NodeManagerAddress"]: row["NodeID"] for row in ray.nodes() if row["Alive"]}
            rows = ray.get([run_probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                nodes[ip], soft=False)).remote(role, rank)
                for ip, role, rank in HOSTS])
            (BASE / "mooncake-qualification.json").write_text(json.dumps(rows, indent=2) + "\n")
            for row in rows:
                print(json.dumps({key: row[key] for key in ("ip", "role", "code", "timed_out", "events", "output_tail")}), flush=True)
                assert row["code"] == 0 and not row["timed_out"]
                assert not row["gpu_processes_after"] and row["closed"]
                assert [event["version"] for event in row["events"]] == list(range(-1, 3))
                assert all(event["verified"] and event["register_calls"] == 2 for event in row["events"])
                assert len({tuple(e["registered_addresses"]) for e in row["events"]}) == 1
                assert any(row["after"][name][key] > row["before"][name][key]
                           for name in row["before"] for key in ("port_xmit_data", "port_rcv_data"))
            assert len({row["artifact"]["awex_transport_sha256"] for row in rows}) == 1
            assert len({row["artifact"]["awex_revision"] for row in rows}) == 1
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
