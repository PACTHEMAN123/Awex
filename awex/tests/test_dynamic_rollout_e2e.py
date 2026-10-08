"""Portable, real late-process join coverage (CUDA execution is mocked)."""

import json
import subprocess
import sys
from pathlib import Path


def test_late_rollout_processes_receive_the_immediate_next_fp8_update(tmp_path):
    output = tmp_path / "results.json"
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
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(output.read_text())
    assert summary["passed"] and not summary["cuda_kernel_validated"]
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
