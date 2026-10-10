"""Choose recipe GPUs across local RDMA ports, retaining replica locality."""

import os
import re
import subprocess

from shardstream.topology import _active_rdma_endpoints, _parse_nvidia_topology


def _cpu_list(value):
    cpus = set()
    for part in value.split(","):
        bounds = [int(v) for v in part.split("-")]
        cpus.update(range(bounds[0], bounds[-1] + 1))
    return cpus


def plan_nic_spread(
    gpus, count, replica_tp=1, hca_span=1, *, topology=None, endpoints=None
):
    """Return selected physical GPUs and an explicit per-GPU HCA/CPU plan."""
    if count < 1 or count > len(gpus):
        raise ValueError("Requested GPUs exceed this actor's reservation")
    endpoints = _active_rdma_endpoints() if endpoints is None else endpoints
    if not endpoints:
        raise ValueError("NIC spread requires active RDMA ports")
    if topology is None:
        topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    distances = _parse_nvidia_topology(topology, endpoints, gpus)
    if distances is None:
        raise ValueError("Cannot resolve reserved GPUs to RDMA topology")
    buckets = [[] for _ in endpoints]
    unassigned = set(range(len(gpus)))
    # First cover every HCA with its closest reserved GPU. Subsequent rounds
    # fill each HCA bucket without concentrating all ranks in one NUMA domain.
    while unassigned:
        for hca in range(len(endpoints)):
            if not unassigned:
                break
            gpu = min(unassigned, key=lambda g: (distances[g][hca], g))
            buckets[hca].append(gpu)
            unassigned.remove(gpu)
    interleaved = [
        b[i] for i in range(max(map(len, buckets))) for b in buckets if i < len(b)
    ]
    # TP2 replica ingress alternates between the two TP ranks. When a whole
    # replica can fit near each HCA, preserve that grouping so its ingress
    # remains distributed. TP1 replicas use interleaved HCA order instead.
    if replica_tp > 1 and count >= len(endpoints) * replica_tp:
        order = [g for b in buckets for g in b]
    else:
        order = interleaved
    selected = order[:count]
    text = re.sub(r"\x1b\[[0-9;]*m", "", topology)
    rows = {
        r[0]: r
        for line in text.splitlines()
        if (r := line.split()) and re.fullmatch(r"GPU\d+", r[0])
    }
    allowed = os.sched_getaffinity(0)
    affinity = {}
    for gpu in selected:
        hca = next(i for i, b in enumerate(buckets) if gpu in b)
        row = rows[f"GPU{gpus[gpu]}"]
        cpus = sorted(_cpu_list(row[-3]) & allowed)
        if not cpus:
            raise ValueError("GPU NUMA domain has no CPUs in the actor's affinity")
        local_hcas = sorted(
            range(len(endpoints)), key=lambda h: (distances[gpu][h], h)
        )[:hca_span]
        if hca not in local_hcas:
            local_hcas[-1] = hca
        affinity[gpus[gpu]] = {
            "hca": "="
            + ",".join(f"{endpoints[h].name}:{endpoints[h].port}" for h in local_hcas),
            "cpus": cpus,
            "numa": int(row[-2]),
            "distance": distances[gpu][hca],
        }
    return [gpus[g] for g in selected], affinity
