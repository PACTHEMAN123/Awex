"""Twenty-update runs and a common window for retained legacy evidence."""
import math
import statistics

NUM_UPDATES = 20
WARMUP_UPDATES = 5
MEASURED_UPDATES = NUM_UPDATES - WARMUP_UPDATES


def launch_protocol(case, outcome=None):
    # Old immutable cases predate explicit protocol fields and requested 35.
    total = case.get('num_updates', 35)
    warmup = case.get('warmup_updates', WARMUP_UPDATES)
    assert total in (20, 35) and warmup == WARMUP_UPDATES
    command = case['command']
    assert int(command[command.index('--num-updates') + 1]) == total
    assert int(command[command.index('--warmup-updates') + 1]) == warmup
    if outcome is not None:
        assert outcome['num_updates'] == total and outcome['warmup_updates'] == warmup
    return total, warmup


def comparison_summary(summary):
    """Return a derived first-20 window; never modify the raw audit summary."""
    total = summary['verified_updates']
    assert total in (20, 35)
    assert summary['measured_updates'] == len(summary['samples_ms']) == total - WARMUP_UPDATES
    samples = summary['samples_ms'][:MEASURED_UPDATES]
    assert len(samples) == MEASURED_UPDATES
    assert all(math.isfinite(v) and v > 0 for v in samples)
    result = dict(summary, raw_verified_updates=total,
        raw_measured_updates=summary['measured_updates'],
        verified_updates=NUM_UPDATES, measured_updates=MEASURED_UPDATES,
        samples_ms=samples, p50_ms=statistics.median(samples),
        p95_ms=sorted(samples)[math.ceil(.95 * len(samples)) - 1],
        comparison_window='first 20 updates: 5 warmup, 15 measured')
    if 'worker_checks' in summary:
        assert summary['worker_checks'] % total == 0
        result.update(raw_worker_checks=summary['worker_checks'],
                      worker_checks=summary['worker_checks'] // total * NUM_UPDATES)
    return result
