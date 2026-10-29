import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

PATH = Path(__file__).resolve().parents[1] / 'benchmarks/experiments/figure10_20261029/protocol.py'
spec = importlib.util.spec_from_file_location('figure10_protocol', PATH)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def fixture_case(tmp_path, config='F', cap=128 * 2**20, pointer_error=False):
    p = protocol
    layout, training = p.LAYOUTS[config]
    case = dict(config=config, repeat=0, label=f'{config}-r0', layout=layout, updates=35, warmup=5,
                revision=p.REVISION, weightbridge_revision=p.WB_REVISION, round_cap_bytes=2**31,
                logical_tensor_cap_bytes=128 * 2**20, backend='weightbridge_native')
    outcome = dict(layout=layout, num_updates=35, warmup_updates=5, publication_mechanism='weightbridge_native',
                   publication_bucket_mb=2048, weight_format='bf16', passed=True, training_hosts=p.IPS[:2],
                   rollout_hosts=p.IPS[2:], training=training,
                   rollout=dict(tp=4, dp=1, ep=1, instances=4, world_size=16),
                   prefix_caching_disabled=True, gpu_placement='nic-spread', transport_environment={
                       'SHARDSTREAM_COMM_MEMORY_PROFILE': '0', 'SHARDSTREAM_DP_SOURCE_POLICY': 'replica',
                       'SHARDSTREAM_REPLICA_ROUTING': 'engine', 'WBRIDGE_LOGICAL_TENSOR_CAP_BYTES': str(128 * 2**20)},
                   generation_before=[{'message': 'ok'}] * 4, generation_after=[{'message': 'ok'}] * 4)
    inventory = [dict(ip=ip, revision=p.REVISION, gpu_processes='') for ip in p.IPS]
    for name, value in [('case.json', case), ('outcome.json', outcome), ('exit.json', {'code': 0}),
                        ('inventory.json', inventory), ('final-inventory.json', inventory)]:
        p.save(tmp_path / name, value)
    records = [dict(version=version, end_to_end_update_ms=version + 1, ranks={
        str(engine): [dict(verified=True, version=version,
                          pointers=[rank + (100 if pointer_error and version == 34 else 0)])
                      for rank in range(4)] for engine in range(4)}) for version in range(35)]
    messages = [('WB_ROUND_CAP', dict(effective_bytes=2**31, requested_bytes=2**31, logical_tensor_cap_bytes=cap)),
                ('WB_CHECKPOINT_GUARD', dict(verified=True)),
                ('WB_SPECGEN_INDEX', dict(preserves_upstream_merge_order=True)),
                ('WB_SOURCE_UNION_INDEX', dict(preserves_upstream_merge_order=True))]
    raw = '\n'.join(marker + ' ' + json.dumps(value) for marker, value in messages for _ in range(32))
    manifest = []
    for index, ip in enumerate(p.IPS):
        path = tmp_path / (ip + '.tar.gz')
        with tarfile.open(path, 'w:gz') as archive:
            if index == 0:
                for name, content in [('training.log', raw), ('training-publications.json', json.dumps(dict(passed=True, records=records)))]:
                    data = content.encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        manifest.append(dict(file=path.name, bytes=path.stat().st_size, sha256=p.hashlib.sha256(path.read_bytes()).hexdigest()))
    p.save(tmp_path / 'manifest.json', manifest)
    return tmp_path


@pytest.mark.parametrize('config', ['F', 'G'])
def test_accepts_complete_protocol(tmp_path, config):
    result = protocol.audit(fixture_case(tmp_path, config))
    assert result['p50_ms'] == 20.5 and result['p95_ms'] == 34
    assert result['rollout_parameter_checks'] == 560


@pytest.mark.parametrize('kwargs', [{'cap': 32 * 2**20}, {'pointer_error': True}])
def test_rejects_cap_or_pointer_drift(tmp_path, kwargs):
    with pytest.raises(AssertionError):
        protocol.audit(fixture_case(tmp_path, **kwargs))


@pytest.mark.parametrize('field,value', [('publication_bucket_mb', 512), ('warmup_updates', 2), ('training_hosts', protocol.IPS[2:])])
def test_rejects_different_experiment(tmp_path, field, value):
    fixture_case(tmp_path)
    outcome = protocol.load(tmp_path / 'outcome.json')
    outcome[field] = value
    protocol.save(tmp_path / 'outcome.json', outcome)
    with pytest.raises(AssertionError):
        protocol.audit(tmp_path)


def test_rejects_archive_corruption(tmp_path):
    fixture_case(tmp_path)
    with next(tmp_path.glob('*.tar.gz')).open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(AssertionError):
        protocol.audit(tmp_path)
