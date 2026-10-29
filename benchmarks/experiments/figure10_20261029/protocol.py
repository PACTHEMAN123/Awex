"""Figure 10 F/G protocol, using Figure 9's native WeightBridge settings."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

BASE = Path('/tmp/weightbridge-fig10-1029')
ASSETS = Path('/tmp/weightbridge-memory-1029')
SOURCE = ASSETS / 'source'
STAGE = ASSETS / 'stage'
IPS = ['11.18.56.89', '11.18.50.148', '11.18.50.220', '11.17.132.185']
REVISION = '8020f16dfff48011428f53948e265d2f62ecc269'
WB_REVISION = '17ea1bc6370424d52152880834a94659db5002ec'
DEPS = f'/tmp/verl-nccl-1020-deps:{ASSETS}/weightbridge:/tmp/comm-memory-1021-mooncake'
LAYOUTS = {
    'F': ('qwen30-tp4-cp2-ep8-r4tp4', dict(tp=4, pp=1, cp=2, ep=8, etp=1, world_size=16)),
    'G': ('qwen30-tp2-pp4-cp2-ep4-r4tp4', dict(tp=2, pp=4, cp=2, ep=4, etp=1, world_size=16)),
}
ORDER = [('F', 0), ('G', 0), ('G', 1), ('F', 1), ('F', 2), ('G', 2)]


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def audit(directory):
    case, outcome = load(directory / 'case.json'), load(directory / 'outcome.json')
    layout, training = LAYOUTS[case['config']]
    assert case['revision'] == REVISION and case['weightbridge_revision'] == WB_REVISION
    assert case['layout'] == outcome['layout'] == layout
    assert case['updates'] == outcome['num_updates'] == 35
    assert case['warmup'] == outcome['warmup_updates'] == 5
    assert case['round_cap_bytes'] == 2**31 and case['logical_tensor_cap_bytes'] == 128 * 2**20
    assert outcome['publication_mechanism'] == case['backend'] == 'weightbridge_native'
    assert outcome['publication_bucket_mb'] == 2048 and outcome['weight_format'] == 'bf16'
    assert outcome['passed'] and load(directory / 'exit.json')['code'] == 0
    assert outcome['training_hosts'] == IPS[:2] and outcome['rollout_hosts'] == IPS[2:]
    assert all(outcome['training'][key] == value for key, value in training.items())
    assert all(outcome['rollout'][key] == value for key, value in
               dict(tp=4, dp=1, ep=1, instances=4, world_size=16).items())
    assert outcome['prefix_caching_disabled'] and outcome['gpu_placement'] == 'nic-spread'
    environment = outcome['transport_environment']
    assert environment['SHARDSTREAM_COMM_MEMORY_PROFILE'] == '0'
    assert environment['SHARDSTREAM_DP_SOURCE_POLICY'] == 'replica'
    assert environment['SHARDSTREAM_REPLICA_ROUTING'] == 'engine'
    assert environment['WBRIDGE_LOGICAL_TENSOR_CAP_BYTES'] == str(128 * 2**20)
    assert len(outcome['generation_before']) == len(outcome['generation_after']) == 4
    assert [row['message'] for row in outcome['generation_before']] == [
        row['message'] for row in outcome['generation_after']]
    for filename in ('inventory.json', 'final-inventory.json'):
        rows = load(directory / filename)
        assert {row['ip'] for row in rows} == set(IPS)
        assert all(not row['gpu_processes'].strip() and row['revision'] == REVISION for row in rows)
    records = None
    events = {key: [] for key in ('WB_ROUND_CAP', 'WB_CHECKPOINT_GUARD',
                                 'WB_SPECGEN_INDEX', 'WB_SOURCE_UNION_INDEX')}
    manifest = load(directory / 'manifest.json')
    assert len(manifest) == 4 and {row['file'] for row in manifest} == {ip + '.tar.gz' for ip in IPS}
    for row in manifest:
        path = directory / row['file']
        assert path.stat().st_size == row['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if member.name == 'training-publications.json':
                    assert records is None
                    data = json.load(archive.extractfile(member))
                    assert data['passed']
                    records = data['records']
                elif member.isfile() and member.name.endswith('.log'):
                    raw = archive.extractfile(member).read().decode(errors='replace')
                    for marker, values in events.items():
                        for match in re.finditer(marker + ' ', raw):
                            values.append(json.JSONDecoder().raw_decode(raw[match.end():])[0])
    assert all(len(values) == 32 for values in events.values()), {k: len(v) for k, v in events.items()}
    assert all(row['effective_bytes'] == row['requested_bytes'] == 2**31 and
               row['logical_tensor_cap_bytes'] == 128 * 2**20 for row in events['WB_ROUND_CAP'])
    assert all(row['verified'] for row in events['WB_CHECKPOINT_GUARD'])
    for marker in ('WB_SPECGEN_INDEX', 'WB_SOURCE_UNION_INDEX'):
        assert all(row['preserves_upstream_merge_order'] for row in events[marker])
    assert len(records) == len({row['version'] for row in records}) == 35
    pointers = {}
    for record in records:
        assert len(record['ranks']) == 4
        for engine, workers in record['ranks'].items():
            assert len(workers) == 4
            for rank, worker in enumerate(workers):
                assert worker['verified'] and worker['version'] == record['version']
                assert pointers.setdefault((engine, rank), worker['pointers']) == worker['pointers']
    samples = [row['end_to_end_update_ms'] for row in records[5:]]
    assert len(samples) == 30 and all(math.isfinite(value) and value > 0 for value in samples)
    return dict(config=case['config'], repeat=case['repeat'], label=case['label'], passed=True,
                revision=REVISION, weightbridge_revision=WB_REVISION, versions=35, measured_updates=30,
                rollout_parameter_checks=560, checkpoint_guards=32, effective_cap_checks=32,
                round_cap_bytes=2**31, logical_tensor_cap_bytes=128 * 2**20,
                p50_ms=statistics.median(samples), p95_ms=sorted(samples)[28], samples_ms=samples)
