from types import SimpleNamespace
import sys

import pytest

from shardstream.integrations.publication import weightbridge_native as wb


def test_round_cap_is_set_before_instrumentation_import(monkeypatch):
    from shardstream.profiling import timeline
    requested = 512 << 20
    def instrument():
        assert wb.os.environ["WBRIDGE_ROUND_CAP_BYTES"] == str(requested)
        monkeypatch.setitem(sys.modules, "wbridge.backend.router",
                            SimpleNamespace(RECEIVER_ROUND_CAP_BYTES=requested))
    monkeypatch.setattr(timeline, "instrument_weightbridge", instrument)
    monkeypatch.delenv("WBRIDGE_ROUND_CAP_BYTES", raising=False)
    wb._configure(requested)


def test_stale_imported_round_cap_fails_instead_of_mislabeling(monkeypatch):
    from shardstream.profiling import timeline
    monkeypatch.setattr(timeline, "instrument_weightbridge", lambda: None)
    monkeypatch.setitem(sys.modules, "wbridge.backend.router",
                        SimpleNamespace(RECEIVER_ROUND_CAP_BYTES=2 << 30))
    monkeypatch.delenv("WBRIDGE_ROUND_CAP_BYTES", raising=False)
    with pytest.raises(RuntimeError, match="imported with round cap"):
        wb._configure(512 << 20)


def test_parameter_views_allow_probe_mutation_without_copy():
    model = wb.torch.nn.Linear(3, 2, bias=False)
    view = wb._parameter_views(model)["weight"]
    assert not view.requires_grad
    assert view.data_ptr() == model.weight.data_ptr()
    view.zero_()
    assert wb.torch.count_nonzero(model.weight).item() == 0
    assert model.weight.requires_grad


def test_probe_guard_rejects_wrong_restore_even_if_later_checks_use_it():
    model = wb.torch.nn.Linear(3, 2, bias=False)
    with pytest.raises(RuntimeError, match="probe changed checkpoint parameters"):
        with wb._model_probe_guard(model, "rollout", 0):
            # A self-consistent restore through a wrong map is still wrong.
            model.weight.data.copy_(model.weight.detach().flip(1))


def test_probe_guard_allows_native_loader_storage_replacement(capsys):
    model = wb.torch.nn.Linear(3, 2, bias=False)
    original = model.weight
    with wb._model_probe_guard(model, "rollout", 0):
        model.weight = wb.torch.nn.Parameter(model.weight.detach().clone())
    assert model.weight.data_ptr() != original.data_ptr()
    assert '"verified": true' in capsys.readouterr().out


def test_probe_guard_preserves_original_specgen_error():
    model = wb.torch.nn.Linear(3, 2, bias=False)
    with pytest.raises(ValueError, match="specgen failed"):
        with wb._model_probe_guard(model, "training", 0):
            model.weight.data.zero_()
            raise ValueError("specgen failed")


def test_close_before_receiver_initialization_does_not_mask_failure():
    wb.WeightBridgePublication(SimpleNamespace(is_driver=True)).close()


@pytest.mark.parametrize("previous", [None, "True"])
def test_independent_rendezvous_restores_environment_on_error(monkeypatch, previous):
    key = "TORCHELASTIC_USE_AGENT_STORE"
    if previous is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, previous)
    with pytest.raises(RuntimeError):
        with wb._independent_rendezvous():
            assert wb.os.environ[key] == "False"
            raise RuntimeError("probe")
    assert wb.os.environ.get(key) == previous


def test_factory_state_is_lazy_and_recallable():
    calls = []
    def factory():
        calls.append(1)
        return object()
    source = wb.ProbeSource(SimpleNamespace(config="original"), {"weight": factory})
    assert list(source.state) == ["weight"] and len(source.state) == 1
    assert source.state.source.get_all_keys() == ("weight",)
    assert "weight" in source.state and "missing" not in source.state
    assert not calls
    assert source.state["weight"] is not source.state["weight"]
    assert len(calls) == 2
    assert source.config == "original"
    with pytest.raises(KeyError):
        source.state["missing"]


def test_publisher_waits_for_consumption_not_just_pack_event(monkeypatch):
    events = []
    harness = SimpleNamespace(is_driver=False, _training_barrier=lambda: events.append("barrier"))
    publisher = wb.WeightBridgePublication(harness)
    publisher.adapter = SimpleNamespace(
        connect=lambda: events.append("connect"),
        send_weights=lambda: SimpleNamespace(synchronize=lambda: events.append("pack_complete")),
        wait_send_complete=lambda: events.append("consumed"),
        flush_profile_outputs=lambda: events.append("flush"),
    )
    publisher.publish()
    publisher.publish()
    assert events == ["barrier", "connect", "pack_complete", "consumed", "flush",
                      "barrier", "pack_complete", "consumed", "flush"]


def test_receiver_polls_until_actual_consumption(monkeypatch):
    events = []
    poll = iter((False, False, True))
    receiver = wb.WeightBridgeReceiver({"timeout_seconds": 10}, 2)
    receiver.adapter = SimpleNamespace(poll_requests=lambda: next(poll),
                                      flush_profile_outputs=lambda: events.append("flush"))
    monkeypatch.setattr(wb.torch.cuda, "synchronize", lambda: events.append("sync"))
    assert receiver.update(None, 5) == {"worker_rank": 2, "step_id": 5, "consumed": True}
    assert events == ["sync", "flush"]


def test_receiver_timeout_is_not_reported_as_consumed():
    receiver = wb.WeightBridgeReceiver({"timeout_seconds": 0}, 0)
    with pytest.raises(TimeoutError):
        receiver.update(None, 0)


def test_vllm_loader_probe_includes_native_post_load_weights(monkeypatch):
    class Model(wb.torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = wb.torch.nn.Parameter(wb.torch.zeros(2, 2))
            self.derived = wb.torch.nn.Parameter(wb.torch.zeros(2, 2))
        def load_weights(self, weights):
            self.weight.copy_(dict(weights)["weight"])
    model = Model()
    config = SimpleNamespace(dtype=wb.torch.float32)
    calls = []
    def post_load(actual_model, actual_config, device):
        assert actual_model is model and actual_config is config
        assert device == model.weight.device
        calls.append("post_load")
        model.derived.copy_(model.weight.T)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.utils",
                        SimpleNamespace(process_weights_after_loading=post_load))
    weights = wb.torch.arange(4, dtype=wb.torch.float32).reshape(2, 2)
    wb._load_vllm_weights(SimpleNamespace(model=model, model_config=config),
                          {"weight": lambda: weights})
    assert calls == ["post_load"]
    assert wb.torch.equal(model.weight, weights)
    assert wb.torch.equal(model.derived, weights.T)


def test_fused_checkpoint_probe_views_preserve_native_shapes_values_and_storage():
    source = wb.torch.arange(24, dtype=wb.torch.float32).reshape(2, 3, 4)
    vector = wb.torch.arange(4, dtype=wb.torch.float32)
    model = wb.torch.nn.Linear(4, 3)
    received = []
    def loader(factories):
        received.append({name: factory() for name, factory in factories.items()})
    factories, shapes, views, load = wb._normalize_probe_views(
        {"fused": lambda: source, "vector": lambda: vector},
        {"fused": (2, 3, 4), "vector": (4,)}, model, loader)
    keys = ["fused::wb_matrix[0]", "fused::wb_matrix[1]"]
    assert shapes == {keys[0]: (3, 4), keys[1]: (3, 4), "vector": (4,)}
    for i, key in enumerate(keys):
        assert wb.torch.equal(factories[key](), source[i])
        assert factories[key]().data_ptr() == source[i].data_ptr()
    probe = wb.torch.arange(24, dtype=wb.torch.float32).reshape(2, 3, 4) + 100
    load({keys[i]: (lambda i=i: probe[i]) for i in range(2)} | {"vector": lambda: vector})
    assert received[0]["fused"].shape == source.shape
    assert set(received[0]) == {"fused", "vector"}
    assert wb.torch.equal(received[0]["fused"], probe)
    with pytest.raises(ValueError, match="Incomplete checkpoint matrix probes"):
        load({keys[0]: lambda: probe[0]})
    actual = views()
    assert actual["bias"].shape == (1, 3)
    assert actual["bias"].data_ptr() == model.bias.data_ptr()
    actual["bias"].fill_(7)
    assert wb.torch.equal(model.bias, wb.torch.full_like(model.bias, 7))
    assert actual["weight"].data_ptr() == model.weight.data_ptr()


def test_matrix_checkpoint_keeps_existing_probe_contract():
    source = wb.torch.zeros(3, 4)
    model = wb.torch.nn.Linear(4, 3)
    factories = {"weight": lambda: source}
    shapes = {"weight": (3, 4)}
    loader = lambda factories: None
    actual, actual_shapes, views, actual_loader = wb._normalize_probe_views(
        factories, shapes, model, loader)
    assert actual is factories and actual_shapes is shapes and actual_loader is loader
    assert views()["bias"].shape == model.bias.shape


def test_storage_probe_uses_full_owner_for_strided_mla_aliases():
    model = wb.torch.nn.Module()
    model.weight = wb.torch.nn.Parameter(wb.torch.arange(24.).reshape(6, 4))
    latent = model.weight.T.view(4, 2, 3)
    uk, uv = latent.split([1, 2], dim=-1)
    model.W_UV = wb.torch.nn.Parameter(uv.transpose(0, 1))
    model.W_UK_T = wb.torch.nn.Parameter(uk.permute(1, 2, 0))
    assert not model.W_UV.is_contiguous()
    actual = wb._storage_parameter_views(model)
    assert set(actual) == {"weight"}
    assert actual["weight"].data_ptr() == model.weight.data_ptr()
    actual["weight"].add_(100)
    assert wb.torch.equal(model.W_UV, (model.weight.T.view(4, 2, 3)[..., 1:]).transpose(0, 1))
    assert wb.torch.equal(model.W_UK_T, model.weight.T.view(4, 2, 3)[..., :1].permute(1, 2, 0))


def test_dense_transposed_storage_view_shares_native_values():
    model = wb.torch.nn.Module()
    model.weight = wb.torch.nn.Parameter(wb.torch.arange(12.).reshape(3, 4).T)
    actual = wb._storage_parameter_views(model)["weight"]
    assert actual.is_contiguous() and actual.data_ptr() == model.weight.data_ptr()
    actual.fill_(9)
    assert wb.torch.equal(model.weight, wb.torch.full_like(model.weight, 9))


def test_strided_uncovered_storage_is_rejected_before_probe_mutation():
    model = wb.torch.nn.Module()
    model.weight = wb.torch.nn.Parameter(wb.torch.arange(20.).reshape(4, 5)[:, 1:4])
    original = model.weight.detach().clone()
    with pytest.raises(ValueError, match="without a full owner"):
        wb._storage_parameter_views(model)
    assert wb.torch.equal(model.weight, original)


def test_expert_matrix_probe_boundaries_survive_independent_transposes():
    source = wb.torch.arange(60.).reshape(3, 4, 5)
    model = wb.torch.nn.Module()
    model.experts = wb.torch.nn.Parameter(source.transpose(1, 2).contiguous())
    def native_loader(factories):
        model.experts.data.copy_(factories["experts"]().transpose(1, 2))
    factories, shapes, views, load = wb._normalize_probe_views(
        {"experts": lambda: source}, {"experts": source.shape}, model, native_loader)
    expected = model.experts.detach().clone()
    probes = {k: (lambda v=f(): v + 1000) for k, f in factories.items()}
    load(probes)
    assert wb.torch.equal(model.experts, expected + 1000)
    # Restore from the native-shaped sources; matrices have their own identities.
    load(factories)
    assert wb.torch.equal(model.experts, expected)
    assert all(tuple(shape) == (4, 5) for shape in shapes.values())
    assert views()["experts"].data_ptr() == model.experts.data_ptr()


def test_checkpoint_views_include_persistent_router_bias_but_not_runtime_counters():
    model = wb.torch.nn.Module()
    model.router = wb.torch.nn.Module()
    model.router.register_buffer("expert_bias", wb.torch.arange(4.))
    model.router.register_buffer("local_tokens", wb.torch.ones(4), persistent=False)
    actual = wb._parameter_views(model)
    assert set(actual) == {"router.expert_bias"}
    assert actual["router.expert_bias"].data_ptr() == model.router.expert_bias.data_ptr()
    with pytest.raises(RuntimeError, match="probe changed checkpoint parameters"):
        with wb._model_probe_guard(model, "training", 0):
            model.router.expert_bias.add_(1)
