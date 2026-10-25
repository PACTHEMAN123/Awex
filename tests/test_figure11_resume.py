import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "figure11_plan", Path(__file__).parents[1] / "benchmarks/experiments/figure11_20261025/plan.py"
)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def test_rdma_completion_does_not_skip_nccl_replacement(tmp_path):
    row = dict(case_id="inter-r1", backend="awex", repeat=0, mechanism="awex_native_nccl")
    old = tmp_path / "inter-r1-awex-r0-attempt5"
    old.mkdir()
    (old / "complete.json").write_text('{"passed": true}')
    (old / "case.json").write_text(json.dumps(dict(row, mechanism="awex_native_rdma")))
    assert plan.matching_completions(tmp_path, row) == []
    new = tmp_path / "inter-r1-awex-r0-attempt6"
    new.mkdir()
    (new / "complete.json").write_text('{"passed": true}')
    (new / "case.json").write_text(json.dumps(row))
    assert plan.matching_completions(tmp_path, row) == [new / "complete.json"]
