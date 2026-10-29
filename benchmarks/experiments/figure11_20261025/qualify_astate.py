"""Reserve two whole GPU nodes and verify official AState with native RDMA."""
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
HOSTS = [("11.18.50.148", "send"), ("11.18.50.220", "receive")]


@ray.remote(num_cpus=1, num_gpus=8)
def run_probe(role, peer):
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
    manifest = json.loads((BASE / "astate-artifact.json").read_text())
    for name, checksum in manifest["files"].items():
        assert hashlib.sha256((BASE / "astate-runtime" / name).read_bytes()).hexdigest() == checksum
    assert not gpu_processes(), "node already has active GPU processes"
    observer = BASE / "astate_registration_observer.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-std=c11",
                    str(DIRECTORY / "astate_registration_observer.c"), "-ldl", "-o", str(observer)], check=True)
    env = dict(os.environ, PYTHONPATH=str(BASE / "stage") + ":" + manifest["pythonpath"],
               LD_LIBRARY_PATH=manifest["ld_library_path"], LD_PRELOAD=str(observer),
               SHARDSTREAM_ASTATE_COUNTER_EXTENSION=manifest["extension"])
    env.pop("UTRANS_DISABLE_RDMA", None)
    before = network()
    command = ["/tmp/verl-nccl-1020-runtime/bin/python", "-u", str(DIRECTORY / "astate_probe.py"),
               role, "--local-ip", ip, "--peer-ip", peer, "--counter-library", str(observer)]
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
    log = BASE / f"astate-probe-{role}.log"
    log.write_bytes(output)
    for _ in range(30):
        active = gpu_processes()
        if not active:
            break
        time.sleep(1)
    text = output.decode(errors="replace")
    events = [json.loads(line) for line in text.splitlines() if line.startswith('{"event": "ASTATE_PROBE"')]
    return dict(ip=ip, role=role, code=process.returncode, timed_out=timed_out,
                command=command, log=str(log), log_sha256=hashlib.sha256(output).hexdigest(),
                events=events, before=before, after=after, gpu_processes_after=active,
                artifact=manifest, output_tail=text[-5000:])


def main():
    with open("/mnt/fuse/verl-e2e/runs/five-model-baselines-20261012/matrix/active.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ray.init(address="11.18.56.89:6381", logging_level="ERROR")
        try:
            nodes = {row["NodeManagerAddress"]: row["NodeID"] for row in ray.nodes() if row["Alive"]}
            rows = ray.get([run_probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                nodes[ip], soft=False)).remote(role, HOSTS[1 - index][0])
                for index, (ip, role) in enumerate(HOSTS)])
            (BASE / "astate-qualification.json").write_text(json.dumps(rows, indent=2) + "\n")
            for row in rows:
                print(json.dumps({key: row[key] for key in ("ip", "role", "code", "timed_out", "events", "output_tail")}), flush=True)
                assert row["code"] == 0 and not row["timed_out"]
                assert not row["gpu_processes_after"]
                assert [event["version"] for event in row["events"]] == list(range(-1, 3))
                assert all(event["verified"] for event in row["events"])
                assert any(row["after"][name][key] > row["before"][name][key]
                           for name in row["before"] for key in ("port_xmit_data", "port_rcv_data"))
            assert len({row["artifact"]["extension_sha256"] for row in rows}) == 1
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
