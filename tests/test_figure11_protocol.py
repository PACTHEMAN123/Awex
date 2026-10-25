import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('figure11_protocol',
    Path(__file__).parents[1] / 'benchmarks/experiments/figure11_20261025/protocol.py')
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


@pytest.mark.parametrize('total', [20, 35])
def test_declared_command_and_outcome_must_match(total):
    case = dict(command=['--num-updates', str(total), '--warmup-updates', '5'])
    if total == 20:
        case.update(num_updates=20, warmup_updates=5)
    assert protocol.launch_protocol(case, dict(num_updates=total, warmup_updates=5)) == (total, 5)
    with pytest.raises(AssertionError):
        protocol.launch_protocol(case, dict(num_updates=19, warmup_updates=5))
    case['command'][1] = '19'
    with pytest.raises(AssertionError):
        protocol.launch_protocol(case)


def test_common_window_keeps_raw_evidence_and_equal_launch_weight():
    prefix = list(range(1, 16))
    legacy = dict(verified_updates=35, measured_updates=30,
        samples_ms=prefix + [10000] * 15, worker_checks=35 * 16)
    short = dict(verified_updates=20, measured_updates=15,
        samples_ms=prefix.copy(), worker_checks=20 * 16)
    old, new = map(protocol.comparison_summary, [legacy, short])
    for field in ['p50_ms', 'p95_ms', 'samples_ms', 'verified_updates', 'measured_updates', 'worker_checks']:
        assert old[field] == new[field]
    assert old['raw_verified_updates'] == 35 and new['raw_verified_updates'] == 20
    assert legacy['samples_ms'] == prefix + [10000] * 15
    assert legacy['worker_checks'] == 35 * 16
    with pytest.raises(AssertionError):
        protocol.comparison_summary(dict(short, samples_ms=prefix[:-1]))
