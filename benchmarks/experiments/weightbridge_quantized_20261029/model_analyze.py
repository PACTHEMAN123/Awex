"""Audit downloaded real-model publications against the existing FP8 baseline."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile


def read(path):
    return json.loads(path.read_text())


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def analyze(directory, reference):
    outcome = read(directory / 'outcome.json')
    assert (directory / 'exit_code').read_text().strip() == '0'
    assert outcome['passed'] and outcome['publication_mechanism'] == 'weightbridge_gather_fp8'
    assert read(directory / 'active.json')['state'] == 'complete'
    for key in ['layout', 'training', 'rollout', 'training_hosts', 'rollout_hosts',
                'node_placements', 'gpu_placement', 'prefix_caching_disabled',
                'rollout_gpu_memory_utilization_override', 'weight_format']:
        assert outcome[key] == reference[key], key
    assert outcome['generation_baseline_after_checkpoint_bootstrap']
    assert outcome['generation_paused_for_publications']
    assert outcome['generation_pause_results'] == [{'status': 'paused'}] * 4
    assert outcome['generation_resume_results'] == [{'status': 'resumed'}] * 4
    assert [v['message'] for v in outcome['generation_before']] == [
        v['message'] for v in outcome['generation_after']]
    publication = None
    prepares = {}
    plans = {}
    mode = read(directory / 'case.json').get('preparation_mode', 'eager')
    manifest = read(directory / 'manifest.json')
    assert {row['ip'] for row in manifest} == set(outcome['training_hosts'] + outcome['rollout_hosts'])
    decoder = json.JSONDecoder()
    for item in manifest:
        assert not item['gpu_processes']
        path = directory / item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        with tarfile.open(path) as archive:
            if 'training-publications.json' in archive.getnames():
                assert publication is None
                publication = json.load(archive.extractfile('training-publications.json'))
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith('.log'):
                    continue
                log = archive.extractfile(member).read().decode(errors='replace')
                for match in re.finditer('WB_FP8_PIPELINE_PLAN ', log):
                    plan, _ = decoder.raw_decode(log[match.end():])
                    assert plan['rank'] not in plans
                    plans[plan['rank']] = plan
                for match in re.finditer('WB_FP8_PREPARE ', log):
                    row, _ = decoder.raw_decode(log[match.end():])
                    key = row['rank'], row['step']
                    assert key not in prepares, key
                    prepares[key] = row
    assert publication and publication['passed'] and publication['real_model_weights']
    rows = publication['records']
    assert len(rows) == outcome['num_updates']
    assert [row['version'] for row in rows] == list(range(len(rows)))
    bootstrap = publication['checkpoint_bootstrap']
    assert bootstrap['version'] == -1
    assert set(prepares) == {(rank, step) for rank in range(16) for step in range(-1, len(rows))}
    pointers, checks = {}, 0
    for row in [bootstrap, *rows]:
        assert len(row['ranks']) == 4
        for engine, workers in row['ranks'].items():
            assert len(workers) == 2
            for index, worker in enumerate(workers):
                assert worker['verified'] and worker['version'] == row['version']
                assert worker['full_checkpoint_reference']
                assert worker['fp8_rounding_mode'] == 'fp32_interval'
                assert worker['fp8_relative_bound'] <= 4 * 2**-23
                assert worker['scale_rtol'] == 5e-7 and worker['scale_atol'] == 0
                assert worker['fp8_elements'] > 0 and worker['scale_elements'] > 0
                assert worker['bf16_fp32_elements'] > 0
                assert worker['parameter_count'] == len(worker['pointers']) > 0
                key = engine, index
                assert pointers.setdefault(key, worker['pointers']) == worker['pointers']
                checks += 1
    measured = rows[outcome['warmup_updates']:]
    samples = [row['end_to_end_update_ms'] for row in measured]
    assert samples and all(math.isfinite(value) and value > 0 for value in samples)
    phases = []
    if mode == 'pipeline':
        assert set(plans) == set(range(16))
        assert len({plan['global_plan_sha256'] for plan in plans.values()}) == 1
        assert all(plan['native_wire_buffers'] == 2 and plan['native_rounds'] > 1 for plan in plans.values())
    overlap_versions, overlap_ranks = set(), set()
    for row in measured:
        ranks = [prepares[rank, row['version']] for rank in range(16)]
        if mode == 'pipeline':
            for rank in ranks:
                assert rank['preparation_mode'] == 'pipeline'
                assert rank['full_tensor_quantizations'] == 2544 and rank['gathered_tensors'] == 192
                assert rank['logical_trainer_gather_egress_bytes'] == 905969664
                recomputed = []
                for current, start, end in rank['preparation_intervals_ns']:
                    for previous, (sent, done) in rank['native_rdma_intervals_ns'].items():
                        duration = min(end, done) - max(start, sent)
                        if int(previous) < current and duration > 0:
                            recomputed.append(dict(preparing_round=current, transmitting_round=int(previous),
                                                   overlap_ms=duration / 1e6))
                assert recomputed == rank['overlap_pairs']
                if recomputed:
                    overlap_versions.add(row['version'])
                    overlap_ranks.add(rank['rank'])
        phases.append(dict(version=row['version'],
            max_source_prepare_ms=max(rank['prepare_ms'] for rank in ranks),
            logical_trainer_gather_egress_bytes=sum(rank['logical_trainer_gather_egress_bytes'] for rank in ranks),
            full_tensor_quantizations=sum(rank['full_tensor_quantizations'] for rank in ranks),
            gathered_tensors=sum(rank['gathered_tensors'] for rank in ranks)))
    if mode == 'pipeline':
        assert overlap_versions == {row['version'] for row in measured}, 'No measured overlap on some updates'
    result = dict(label=directory.name, revision=read(directory / 'case.json')['revision'],
        preparation_mode=mode, measured_versions_with_overlap=sorted(overlap_versions),
        source_ranks_with_overlap=sorted(overlap_ranks), native_round_plans=list(plans.values()),
        p50_ms=statistics.median(samples), p95_ms=percentile(samples, .95), samples_ms=samples,
        verified_receiver_checks=checks, bootstrap_ms=bootstrap['end_to_end_update_ms'],
        num_updates=len(rows), warmup_updates=outcome['warmup_updates'],
        max_source_prepare_p50_ms=statistics.median(p['max_source_prepare_ms'] for p in phases),
        phases=phases, matched_reference_configuration=True,
        prepare_timing=('sum_host_enqueue_intervals' if mode == 'pipeline' else 'eager_preparation_wall_time'),
        scope=('Native-round preparation overlapping background RDMA; persistent full staging; no optimizer steps.'
               if mode == 'pipeline' else 'Eager full-model gather/quantize then unchanged native WeightBridge; no overlap or optimizer steps.'))
    (directory / 'audited-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-outcome', type=Path, required=True)
    parser.add_argument('directories', type=Path, nargs='+')
    args = parser.parse_args()
    reference = read(args.reference_outcome)
    assert reference['passed'] and reference['weight_format'] == 'fp8'
    results = [analyze(directory, reference) for directory in args.directories]
    print(json.dumps(results, indent=2))
