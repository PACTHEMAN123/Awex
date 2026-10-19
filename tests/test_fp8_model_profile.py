"""Reject receiver corruption against an independent, real-weight FP8 oracle."""

import pytest
import torch
from shardstream.profiling.fp8_model import FP8ModelWeightProfile
from shardstream.profiling.model import version_marker_factor


def profile(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    source = torch.linspace(-2, 3, 128 * 256).reshape(128, 256).bfloat16()
    parameters = {
        "layer.weight": torch.empty_like(source, dtype=torch.float8_e4m3fn),
        "layer.weight_scale_inv": torch.empty(1, 2),
        "model.norm.weight": torch.empty(8, dtype=torch.bfloat16),
        "router.weight": torch.empty(4, dtype=torch.float32),
    }
    references = {
        "layer.weight": source,
        "model.norm.weight": torch.linspace(0.5, 1, 8).bfloat16(),
        "router.weight": torch.tensor([0.003127, -1.23117, 2.72327, -0.33373]),
    }
    return FP8ModelWeightProfile(parameters, references, "scale", 35)


def restore(oracle, version):
    for name, actual in oracle.parameters.items():
        if name in oracle.quantized:
            source = oracle.references[name]
            scale = oracle.references[oracle.quantized[name]]
            blocks = source.float().reshape(1, 128, 2, 128)
            actual.copy_((blocks * (1 / scale.double()).float()[:, None, :, None])
                         .reshape(source.shape).to(torch.float8_e4m3fn))
        else:
            actual.copy_(oracle.references[name])
    oracle.parameters[oracle.norm_name].mul_(version_marker_factor(version - 1, oracle.marker_period))


def test_fp8_byte_scale_mixed_dtype_and_stale_version_corruption(monkeypatch):
    oracle = profile(monkeypatch)
    restore(oracle, 0)
    result = oracle.command("verify", 0)
    assert result['fp8_bytes_exact'] and result['fp8_elements'] == 128 * 256
    assert result['scale_elements'] == 2 and result['parameter_count'] == 4
    actual = oracle.parameters['layer.weight'].view(torch.uint8)
    actual[1, 1] ^= 1
    with pytest.raises(AssertionError, match="FP8 byte mismatch"):
        oracle.command("verify", 0)
    restore(oracle, 0)
    oracle.parameters['layer.weight_scale_inv'].mul_(1.01)
    with pytest.raises(AssertionError, match="scale_inv"):
        oracle.command("verify", 0)
    restore(oracle, 0)
    oracle.parameters['router.weight'].copy_(oracle.references['router.weight'].bfloat16().float())
    with pytest.raises(AssertionError, match="router"):
        oracle.command("verify", 0)
    restore(oracle, 0)
    with pytest.raises(AssertionError, match="norm"):
        oracle.command("verify", 1)
    oracle.command("clear", 0)
    assert all(torch.count_nonzero(p.view(torch.uint8)) == 0 for p in oracle.parameters.values())
    with pytest.raises(AssertionError):
        oracle.command("verify", 0)
    restore(oracle, 34)
    assert oracle.command("verify", 34)['verified']


def test_fp8_oracle_rejects_parameter_replacement_and_partial_reference(monkeypatch):
    oracle = profile(monkeypatch)
    oracle.parameters['layer.weight'] = oracle.parameters['layer.weight'].clone()
    with pytest.raises(AssertionError, match="replaced"):
        oracle.command("clear", 0)
    oracle = profile(monkeypatch)
    del oracle.references['router.weight']
    with pytest.raises(ValueError, match="cover every"):
        FP8ModelWeightProfile(oracle.parameters, oracle.references)


def test_fp8_checkpoint_oracle_loads_canonical_shard_and_does_not_clone_dummy(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    safetensors = pytest.importorskip("safetensors.torch")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    tensor = torch.arange(256 * 256).reshape(256, 256).bfloat16() / 1024
    norm = torch.ones(8, dtype=torch.bfloat16)
    safetensors.save_file({'layer.weight': tensor, 'model.norm.weight': norm}, tmp_path/'weights.safetensors')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'layer.weight': 'weights.safetensors', 'model.norm.weight': 'weights.safetensors'}}))
    parameters = {'layer.weight': torch.zeros((128, 256), dtype=torch.float8_e4m3fn),
                  'layer.weight_scale_inv': torch.zeros(1, 2), 'model.norm.weight': torch.zeros_like(norm)}
    metadata = [SimpleNamespace(name=name, shards=[SimpleNamespace(shape=tuple(p.shape),
                global_offset=(128, 0) if name == 'layer.weight' else (0,) * p.ndim)])
                for name, p in parameters.items()]
    reader = SimpleNamespace(parameters=parameters, current_worker_parameters_meta=metadata)
    oracle = FP8ModelWeightProfile.from_checkpoint(reader, tmp_path, 'scale', 35)
    assert torch.equal(oracle.references['layer.weight'], tensor[128:])
    assert not torch.count_nonzero(parameters['layer.weight'].view(torch.uint8))
    with pytest.raises(AssertionError):
        oracle.command('verify', -1)


def test_fp8_server_configuration_is_explicit_and_does_not_require_driver_environment(monkeypatch):
    import json

    from shardstream_benchmarks.fp8_config import fp8_server_args

    monkeypatch.setenv('SHARDSTREAM_FP8_BLOCKWISE', '0')
    args = fp8_server_args(enabled=True, block_shape=(128, 128))
    assert args[args.index('--load-format') + 1] == 'dummy'
    config = json.loads(args[args.index('--hf-overrides') + 1])['quantization_config']
    assert config['weight_block_size'] == [128, 128]
    assert fp8_server_args(enabled=False) == []


@pytest.mark.parametrize('updates', [3, 35])
def test_fp8_bootstrap_and_regular_updates_have_valid_sequences_and_restore_checkpoint(monkeypatch, updates):
    from shardstream.transport import _sequence_from_step

    oracle = profile(monkeypatch)
    oracle.marker_period = updates
    versions = [-1, *range(updates)]
    sequences = [_sequence_from_step(v) for v in versions]
    assert sequences == list(range(1, updates + 2))
    factors = [version_marker_factor(v - 1, updates) for v in versions]
    assert factors[0] == factors[-1] == 1 and len(set(factors[1:])) == updates
    for version in versions:
        restore(oracle, version)
        assert oracle.command('verify', version)['verified']
    assert torch.equal(oracle.parameters[oracle.norm_name], oracle.references[oracle.norm_name])


def test_cold_fp8_clear_precedes_lazy_reader_initialization(monkeypatch):
    from types import SimpleNamespace

    from shardstream.profiling.model import model_profile_task

    monkeypatch.setenv('SHARDSTREAM_FP8_BLOCKWISE', '1')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones((128, 128)).to(torch.float8_e4m3fn),
                                      requires_grad=False)
    scheduler = SimpleNamespace()
    result = model_profile_task('clear', -1, marker_mode='scale', marker_period=35,
                                model=model, model_context={'scheduler': scheduler})
    assert result['cold_native_fp8_storage']
    assert not torch.count_nonzero(model.weight.view(torch.uint8))
    assert scheduler._shardstream_fp8_initial_pointers == {'weight': model.weight.data_ptr()}
