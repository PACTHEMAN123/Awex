import json
import os
from pathlib import Path

import pytest
from shardstream.integrations.affinity import configure_rank_affinity
from shardstream.topology import _RdmaEndpoint
from shardstream_benchmarks.placement import plan_nic_spread


def topology():
    rows = [
        "GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7 NIC0 NIC1 NIC2 NIC3 CPU NUMA GPU_NUMA"
    ]
    nearest = {1: 0, 3: 1, 4: 2, 6: 3}
    for gpu in range(8):
        distances = ["NODE" if (gpu < 4) == (h < 2) else "SYS" for h in range(4)]
        if gpu in nearest:
            distances[nearest[gpu]] = "PIX"
        rows.append(
            f"GPU{gpu} "
            + " ".join(["X"] * 8 + distances)
            + (" 0-3 0 N/A" if gpu < 4 else " 4-7 1 N/A")
        )
    rows.extend(f"NIC{h}: mlx5_{h}" for h in range(4))
    return "\n".join(rows)


@pytest.mark.parametrize(
    "count,tp,expected",
    [
        (4, 1, [1, 3, 4, 6]),
        (8, 1, [1, 3, 4, 6, 0, 2, 5, 7]),
        (8, 2, [1, 0, 3, 2, 4, 5, 6, 7]),
    ],
)
def test_spread_covers_hcas_and_preserves_tp_ingress(count, tp, expected, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    endpoints = [_RdmaEndpoint(f"mlx5_{h}", 1, 400, str(h)) for h in range(4)]
    selected, affinity = plan_nic_spread(
        list(map(str, range(8))), count, tp, topology=topology(), endpoints=endpoints
    )
    assert selected == list(map(str, expected))
    ingress = [0, 3, 4, 7] if tp == 2 else list(range(4))
    assert len({affinity[selected[r]]["hca"] for r in ingress}) == 4
    for gpu, entry in affinity.items():
        assert set(entry["cpus"]) == (
            set(range(4)) if int(gpu) < 4 else set(range(4, 8))
        )


def test_two_ingress_ranks_cover_four_numa_local_hcas(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    endpoints = [_RdmaEndpoint(f"mlx5_{h}", 1, 400, str(h)) for h in range(4)]
    selected, affinity = plan_nic_spread(
        list(map(str, range(8))), 4, 2, 2, topology=topology(), endpoints=endpoints
    )
    ingress = [affinity[selected[r]]["hca"] for r in [0, 3]]
    assert set(",".join(ingress).replace("=", "").split(",")) == {
        f"mlx5_{h}:1" for h in range(4)
    }
    assert all(affinity[g]["distance"] == 0 for g in selected)


def test_rank_affinity_uses_physical_gpu_and_existing_threads(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3,4,6")
    monkeypatch.setenv(
        "SHARDSTREAM_RANK_AFFINITY",
        json.dumps({"6": {"hca": "=mlx5_28:1", "cpus": [4, 5], "numa": 1}}),
    )
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda _: [Path("/proc/self/task/123"), Path("/proc/self/task/124")],
    )
    calls = []
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append((pid, cpus)),
        raising=False,
    )
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {4, 5}, raising=False)
    configure_rank_affinity(3)
    assert calls == [(123, {4, 5}), (124, {4, 5})]
    assert os.environ["NCCL_IB_HCA"] == "=mlx5_28:1"
