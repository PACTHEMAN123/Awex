"""Portable, real late-process join coverage (CUDA execution is mocked)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("node_agents", [False, True])
def test_late_rollout_processes_receive_the_immediate_next_fp8_update(
    tmp_path, node_agents
):
    output = tmp_path / "results.json"
    options = []
    if node_agents:
        placements = tmp_path / "placements.json"
        placements.write_text(
            json.dumps(
                [
                    {"participant": ["training", "", 0], "node": "node-a", "device": 0},
                    {
                        "participant": ["rollout", "engine-0", 0],
                        "node": "node-a",
                        "device": 1,
                    },
                    {
                        "participant": ["rollout", "engine-1", 0],
                        "node": "node-b",
                        "device": 0,
                    },
                    {
                        "participant": ["rollout", "engine-2", 0],
                        "node": "node-b",
                        "device": 1,
                    },
                ]
            )
        )
        options = ["--placements", str(placements), "--local-agents"]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "awex.tests.experimental.nccl_device_v2_dynamic_e2e",
            "--backend",
            "cpu-mock",
            "--fp8",
            "--tp",
            "1",
            "--join-after",
            "2",
            "4",
            "--target-engines",
            "2",
            "3",
            "--updates",
            "6",
            "--timeout",
            "30",
            "--output",
            str(output),
            *options,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(output.read_text())
    assert summary["passed"] and not summary["cuda_kernel_validated"]
    assert summary["worker_launch"] == ("node-agents" if node_agents else "local")
    publications = [
        item for item in summary["records"] if item["event"] == "publication"
    ]
    assert [len(item["engine_ids"]) for item in publications] == [1, 1, 2, 2, 3, 3]
    assert [item["version"] for item in publications] == list(range(6))
    for index, publication in enumerate(publications):
        assert all(
            rank["verified"] and rank["epoch"] == index // 2
            for rank in publication["ranks"].values()
        )
    old = [item["ranks"]["rollout/engine-0/0"] for item in publications]
    assert len({item["pid"] for item in old}) == 1
    assert all(item["pointers"] == old[0]["pointers"] for item in old)
    assert "rollout/engine-1/0" not in publications[1]["ranks"]
    assert "rollout/engine-1/0" in publications[2]["ranks"]
    assert "rollout/engine-2/0" not in publications[3]["ranks"]
    assert "rollout/engine-2/0" in publications[4]["ranks"]
    for record in summary["records"]:
        if record["event"] in ("join", "initial_prepare"):
            for rank in record["ranks"].values():
                assert rank["preparation_metrics"]["device_plan_ready"]
                assert not rank["preparation_metrics"]["kernel_launched"]
        elif record["event"] == "publication":
            for rank in record["ranks"].values():
                assert rank["metrics"]["plan_cache_hit"]
                assert rank["metrics"]["transport_init_time_ms"] == 0
                assert rank["metrics"]["plan_initialization_time_ms"] == 0
