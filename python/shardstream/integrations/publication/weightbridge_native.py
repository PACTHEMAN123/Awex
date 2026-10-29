"""Experimental native WeightBridge frontend integration for BF16 benchmarks.

Framework loaders are probed by WeightBridge itself. No ShardStream layout,
transport, routing, or manually supplied LoadSpec participates in this path.
"""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import json
import os
import time

import torch
import torch.distributed as dist

from shardstream.integrations.publication.registry import (
    PublicationMechanism, publication_endpoints, register_publication_mechanism,
    register_vllm_publication_receiver,
)
from shardstream.integrations.publication.verl_native import _request_endpoints


class FactoryState(Mapping):
    """Expose re-callable WeightBridge factories through Bridge's state mapping."""

    def __init__(self, factories):
        self.factories = factories
        self.source = self

    def get_all_keys(self):
        return tuple(self.factories)

    def __contains__(self, name):
        return name in self.factories

    def __getitem__(self, name):
        return self.factories[name]()

    def __iter__(self):
        return iter(self.factories)

    def __len__(self):
        return len(self.factories)


class ProbeSource:
    def __init__(self, original, factories):
        self.original = original
        self.state = FactoryState(factories)

    def __getattr__(self, name):
        return getattr(self.original, name)


def _configure(round_bytes):
    if int(round_bytes) <= 0:
        raise ValueError("WeightBridge round cap must be positive")
    os.environ["WBRIDGE_ROUND_CAP_BYTES"] = str(int(round_bytes))
    os.environ.setdefault("WBRIDGE_COORDINATOR_REGISTER_TIMEOUT_S", "1200")
    os.environ.setdefault("WBRIDGE_SPECGEN_PROBE_DEVICE", "auto")
    # Importing even gantt initializes the backend package and its round-cap constant.
    from shardstream.profiling.timeline import instrument_weightbridge
    instrument_weightbridge()
    from wbridge.backend.router import RECEIVER_ROUND_CAP_BYTES
    from wbridge.utils.data import logical_tensor_cap_bytes
    if RECEIVER_ROUND_CAP_BYTES != int(round_bytes):
        raise RuntimeError(f"WeightBridge imported with round cap {RECEIVER_ROUND_CAP_BYTES}; requested {round_bytes}")
    print("WB_ROUND_CAP " + json.dumps({"pid": os.getpid(), "requested_bytes": int(round_bytes),
                                      "effective_bytes": RECEIVER_ROUND_CAP_BYTES,
                                      "logical_tensor_cap_bytes": logical_tensor_cap_bytes()}), flush=True)


def _parameter_views(model, *, include_buffers=False):
    """Expose checkpoint parameters and persistent framework buffers."""
    values = {name: param.detach() for name, param in model.named_parameters()}
    if not include_buffers:
        return values
    for prefix, module in model.named_modules():
        for name, value in module.named_buffers(recurse=False):
            if name not in module._non_persistent_buffers_set:
                full_name = f"{prefix}.{name}" if prefix else name
                values[full_name] = value.detach()
    return values


@contextmanager
def _model_probe_guard(model, role, rank):
    """Check specgen restoration against the framework-loaded checkpoint.

    WeightBridge restores and verifies through its inferred LoadSpec. Keep an
    independent reference from before probing, outside publication timing, so
    a wrong mapping cannot become the reference for subsequent update checks.
    """
    # Initialization validation must not leave a full model in the CUDA cache:
    # memory-profile baselines are captured before this independent check.
    expected = {name: value.detach().to(device="cpu", copy=True)
                for name, value in _parameter_views(model, include_buffers=True).items()}
    yield
    actual = _parameter_views(model, include_buffers=True)
    bad = sorted(set(actual) ^ set(expected))
    for name in expected.keys() & actual.keys():
        a, b = actual[name], expected[name]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.detach().cpu(), b):
            bad.append(name)
    if bad:
        raise RuntimeError(f"WeightBridge probe changed checkpoint parameters: {sorted(bad)[:20]} (total={len(bad)})")
    print("WB_CHECKPOINT_GUARD " + json.dumps({
        "pid": os.getpid(), "role": role, "rank": rank, "verified": True,
        "parameter_tensors": len(expected),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in expected.values()),
    }), flush=True)


@contextmanager
def _independent_rendezvous():
    key = "TORCHELASTIC_USE_AGENT_STORE"
    previous = os.environ.get(key)
    os.environ[key] = "False"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _storage_parameter_views(model, *, include_buffers=False):
    """Bind native storage once when strided parameters alias a full owner.

    A reshape of a strided view can allocate a detached copy. Prefer an
    existing contiguous parameter covering its complete storage. Black-box
    probing still discovers every correspondence; no mapping is supplied.
    """
    values = _parameter_views(model, include_buffers=include_buffers)
    def key(value):
        storage = value.untyped_storage()
        return storage.data_ptr(), storage.nbytes(), value.dtype, value.device
    owners = {key(value): name for name, value in values.items()
              if value.is_contiguous() and value.storage_offset() == 0
              and value.numel() * value.element_size() == value.untyped_storage().nbytes()}
    views = {}
    for name, value in values.items():
        if value.is_contiguous():
            views[name] = value
            continue
        if key(value) in owners:
            continue
        # A dense transpose with no registered owner can expose its physical
        # axis order as a shared-storage matrix, without changing the model.
        order = sorted(range(value.ndim), key=lambda axis: -value.stride(axis))
        physical = value.permute(order)
        if not physical.is_contiguous():
            raise ValueError(f"Cannot expose non-dense parameter storage without a full owner: {name}")
        views[name] = physical.view(physical.shape[0], -1)
    return views


def _normalize_probe_views(factories, shapes, model, loader, *, include_buffers=False):
    """Expose each fused HF matrix separately to the native black-box probe.

    Framework loaders receive the original checkpoint shapes and values. The
    probe sees each last-two-axis matrix independently, preserving per-expert
    transposes. All sources and workers expose two matrix axes. No LoadSpec is
    supplied and actual model parameter storage is unchanged.
    """
    import itertools
    import math
    fused = {name: shape for name, shape in shapes.items() if len(shape) > 2}
    if not fused:
        return factories, shapes, lambda: _storage_parameter_views(model, include_buffers=include_buffers), loader
    normalized = dict(factories)
    normalized_shapes = dict(shapes)
    pieces = {}
    for name, shape in fused.items():
        del normalized[name], normalized_shapes[name]
        pieces[name] = []
        # Keep the two matrix axes together. Flattening the expert axis into
        # rows loses per-expert transpose boundaries in a native 2D mapping.
        for index in itertools.product(*(range(n) for n in shape[:-2])):
            key = name + "::wb_matrix[" + ",".join(map(str, index)) + "]"
            if key in normalized:
                raise ValueError(f"Checkpoint matrix-view name collision: {key}")
            pieces[name].append(key)
            normalized_shapes[key] = tuple(shape[-2:])
            def fetch(factory=factories[name], index=index):
                return factory()[index]
            normalized[key] = fetch

    small = {name: shape for name, shape in shapes.items() if len(shape) < 2}
    for name, shape in small.items():
        normalized_shapes[name] = (1, math.prod(shape))
        def fetch(factory=factories[name], shape=normalized_shapes[name]):
            return factory().view(shape)
        normalized[name] = fetch

    def load(probes):
        original = dict(probes)
        for name, shape in small.items():
            if name in probes:
                def fetch(factory=probes[name], shape=shape):
                    return factory().view(shape)
                original[name] = fetch
        for name, shape in fused.items():
            keys = pieces[name]
            if not any(key in probes for key in keys):
                continue
            if not all(key in probes for key in keys):
                raise ValueError(f"Incomplete checkpoint matrix probes for {name}")
            for key in keys:
                del original[key]
            def fetch(keys=tuple(keys), shape=shape):
                return torch.stack([probes[key]() for key in keys]).view(shape)
            original[name] = fetch
        loader(original)

    def worker_views():
        return {name: value.view(1, -1) if value.ndim < 2
                else value.view(-1, value.shape[-1])
                for name, value in _storage_parameter_views(model, include_buffers=include_buffers).items()}
    print("WB_PROBE_MATRIX_VIEWS " + json.dumps({
        "pid": os.getpid(), "fused_sources": len(fused),
        "matrix_sources": sum(map(len, pieces.values())),
        "preserves_expert_matrix_boundaries": True,
        "all_source_and_worker_views_are_matrices": True,
        "checkpoint_values_unchanged": True,
    }), flush=True)
    return normalized, normalized_shapes, worker_views, load


def _normalize_glm_mtp_aliases(factories, shapes, config, loader):
    """Give declared MTP shared embeddings/head one checkpoint identity.

    Megatron uses shared parameters; vLLM's MTP loader reads the checkpoint's
    duplicate names. Check real values before probing, then expose the same
    canonical probe to both native loaders. Never merge arbitrary equal weights.
    """
    if "Glm4MoeLiteForCausalLM" not in config.get("architectures", []):
        return factories, shapes, loader
    aliases = {}
    for offset in range(config.get("num_nextn_predict_layers", 0)):
        layer = config["num_hidden_layers"] + offset
        for alias, canonical in [
            (f"model.layers.{layer}.embed_tokens.weight", "model.embed_tokens.weight"),
            (f"model.layers.{layer}.shared_head.head.weight", "lm_head.weight"),
        ]:
            if alias not in factories:
                continue
            if canonical not in factories or tuple(shapes[alias]) != tuple(shapes[canonical]):
                raise ValueError(f"GLM MTP shared checkpoint alias has incompatible shape: {alias}")
            a, b = factories[alias](), factories[canonical]()
            if a.dtype != b.dtype or not torch.equal(a, b):
                raise ValueError(f"GLM MTP shared checkpoint alias differs from {canonical}: {alias}")
            aliases[alias] = canonical
    if not aliases:
        return factories, shapes, loader
    canonical_factories = {k: v for k, v in factories.items() if k not in aliases}
    canonical_shapes = {k: v for k, v in shapes.items() if k not in aliases}
    def load(probes):
        expanded = dict(probes)
        for alias, canonical in aliases.items():
            if canonical in probes:
                expanded[alias] = probes[canonical]
        loader(expanded)
    print("WB_CHECKPOINT_ALIASES " + json.dumps({
        "pid": os.getpid(), "aliases": aliases,
        "checkpoint_copies_exactly_equal": True,
        "scope": "GLM MTP declared shared embeddings and head",
    }), flush=True)
    return canonical_factories, canonical_shapes, load


def _context(model_path, model, rank, loader, *, include_buffers=False):
    from wbridge.frontend.adapters import AdapterContext
    from wbridge.utils import specgen
    from wbridge.utils.specgen import hf_weights_from_checkpoint
    from shardstream.integrations.publication.weightbridge_specgen import indexed_merge

    # Preserve upstream geometry and first-pair merge ordering, but avoid its
    # quadratic scan over non-adjacent expert-bias fragments. Initialization
    # only: no transfer, routing or buffer configuration is changed.
    if not getattr(specgen._merge_consecutive_mappings, "_indexed_candidates", False):
        adjacent = specgen._merge_adjacent_shards
        def merge(mappings):
            return indexed_merge(mappings, adjacent)
        merge._indexed_candidates = True
        specgen._merge_consecutive_mappings = merge
        print("WB_SPECGEN_INDEX " + json.dumps({"pid": os.getpid(), "rank": rank,
                                               "preserves_upstream_merge_order": True}), flush=True)

    from wbridge.utils import data as weightbridge_data
    from shardstream.integrations.publication.weightbridge_specgen import indexed_source_union
    if not getattr(weightbridge_data._merge_shards_for_src_shard_spec, "_indexed_candidates", False):
        contained = weightbridge_data._shard_contained_in
        adjacent_union = weightbridge_data._try_merge_two_shards
        def source_union(shards):
            return indexed_source_union(shards, contained, adjacent_union)
        source_union._indexed_candidates = True
        weightbridge_data._merge_shards_for_src_shard_spec = source_union
        print("WB_SOURCE_UNION_INDEX " + json.dumps({"pid": os.getpid(), "rank": rank,
            "preserves_upstream_merge_order": True}), flush=True)

    factories, shapes = hf_weights_from_checkpoint(model_path)
    from pathlib import Path
    checkpoint_config = json.loads((Path(model_path) / "config.json").read_text())
    factories, shapes, loader = _normalize_glm_mtp_aliases(
        factories, shapes, checkpoint_config, loader)
    storage_views = _storage_parameter_views(model, include_buffers=include_buffers)
    print("WB_STORAGE_VIEWS " + json.dumps({"pid": os.getpid(), "rank": rank,
        "model_parameters": len(_parameter_views(model, include_buffers=include_buffers)), "storage_views": len(storage_views),
        "strided_aliases_covered_by_owner": len(_parameter_views(model, include_buffers=include_buffers)) - len(storage_views)}), flush=True)
    factories, shapes, worker_views, loader = _normalize_probe_views(
        factories, shapes, model, loader, include_buffers=include_buffers)
    return AdapterContext(factories, shapes, worker_views, loader, rank)


def _native_vllm_model(model_runner):
    """Include the native GLM MTP draft in the same checkpoint target."""
    config = getattr(model_runner, "vllm_config", None)
    if config is None or config.speculative_config is None:
        return model_runner.model
    hf = config.model_config.hf_config
    if "Glm4MoeLiteForCausalLM" not in (getattr(hf, "architectures", None) or []):
        return model_runner.model
    draft = getattr(getattr(model_runner, "drafter", None), "model", None)
    if draft is None:
        raise RuntimeError("GLM MTP checkpoint target is unavailable")
    from shardstream.integrations.models.glm4_moe_lite import GLM47FlashTransferModel
    # This container only exposes existing parameter names and storage. Native
    # vLLM loaders and the black-box probe discover every weight correspondence.
    return GLM47FlashTransferModel(model_runner.model, draft, hf.num_hidden_layers)


def _load_vllm_weights(model_runner, factories):
    """Probe complete native loaders and post-load state for main and MTP."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    model = _native_vllm_model(model_runner)
    targets = [(model_runner.model, model_runner.model_config)]
    if hasattr(model, "target") and hasattr(model, "draft"):
        targets.append((model.draft,
                        model_runner.vllm_config.speculative_config.draft_model_config))
    with torch.no_grad():
        for target, config in targets:
            target.load_weights((name, factory()) for name, factory in factories.items())
            process_weights_after_loading(target, config, next(target.parameters()).device)


@register_vllm_publication_receiver("weightbridge_native")
class WeightBridgeReceiver:
    def __init__(self, config, worker_rank):
        self.config = config
        self.worker_rank = worker_rank
        self.adapter = None
        self.coordinator = None

    def initialize(self):
        _configure(self.config["round_bytes"])
        from wbridge.backend import coordinator
        from wbridge.backend.control_channel import coordinator_ipc, coordinator_tcp_port
        from wbridge.frontend.adapters import ReceiverAdapter

        model = _native_vllm_model(self.model_runner)
        port = self.config["server_port"]
        ipc = coordinator_ipc(port)
        if self.worker_rank == 0:
            self.coordinator = coordinator.spawn(ipc, coordinator_tcp_port(port))

        def load(factories):
            _load_vllm_weights(self.model_runner, factories)

        started = time.perf_counter()
        with _model_probe_guard(model, "rollout", self.worker_rank):
            self.adapter = ReceiverAdapter(
                _context(self.config["model_path"], model, self.worker_rank, load),
                ipc, num_workers=self.config["world_per_engine"],
            )
        torch.cuda.synchronize()
        return {"worker_rank": self.worker_rank, "specgen_ms": (time.perf_counter() - started) * 1000}

    def configure(self, config):
        raise ValueError("WeightBridge connects through its native receiver poll loop")

    def update(self, model, step_id):
        deadline = time.monotonic() + self.config["timeout_seconds"]
        while time.monotonic() < deadline:
            if self.adapter.poll_requests():
                torch.cuda.synchronize()
                self.adapter.flush_profile_outputs()
                return {"worker_rank": self.worker_rank, "step_id": step_id, "consumed": True}
        raise TimeoutError("WeightBridge receiver did not consume the update")

    def close(self):
        if self.adapter is not None:
            self.adapter.receiver.stop()
        if self.coordinator is not None:
            self.coordinator.terminate()
            self.coordinator.wait(timeout=15)
        return {"worker_rank": self.worker_rank, "closed": True}


@register_publication_mechanism("weightbridge_native")
class WeightBridgePublication(PublicationMechanism):
    training_engine_backend = None

    def __init__(self, harness, bucket_size=256 << 20, timeout_seconds=1200, **kwargs):
        super().__init__(harness, **kwargs)
        self.round_bytes = bucket_size
        self.timeout_seconds = timeout_seconds
        self.connected = False
        self.receivers_initialized = False

    def initialize_training(self):
        _configure(self.round_bytes)
        from wbridge.backend.control_channel import coordinator_tcp_port
        from wbridge.backend.sender import SenderArgs
        from wbridge.frontend.adapters import SenderAdapter
        from shardstream._utils.common import get_free_port

        harness = self.harness
        bridge = harness.mcore_bridge
        if bridge is None or not hasattr(bridge, "_model_bridge"):
            raise RuntimeError("WeightBridge requires the loaded Megatron Bridge native loader")
        port = [get_free_port() if harness.is_driver else None]
        if harness.world_size > 1:
            dist.broadcast_object_list(port, src=0)
        endpoints = publication_endpoints(harness)
        args = SenderArgs(
            world_size=harness.world_size,
            receiver_urls=[f"tcp://{host}:{coordinator_tcp_port(p)}" for _, host, p in endpoints],
            master_addr=os.environ["MASTER_ADDR"], master_port=port[0], protocol="rdma",
        )

        def load(factories):
            source = ProbeSource(bridge.hf_pretrained, factories)
            with torch.no_grad():
                bridge._model_bridge.load_weights_hf_to_megatron(source, [harness.mcore_model])

        with _model_probe_guard(harness.mcore_model, "training", harness.rank):
            self.adapter = SenderAdapter(_context(
                harness.inference_config["model_path"], harness.mcore_model, harness.rank, load,
                include_buffers=True), args)
        torch.cuda.synchronize()

    def initialize_driver(self):
        harness = self.harness
        world = int(harness.inference_config["tp_size"]) * int(harness.inference_config.get("dp_size", 1))
        endpoints = publication_endpoints(harness)
        _request_endpoints(endpoints, "publication_init", {
            engine: {"mechanism": self.name, "config": {
                "round_bytes": self.round_bytes, "timeout_seconds": self.timeout_seconds,
                "server_port": port, "world_per_engine": world,
                "model_path": harness.inference_config["model_path"],
            }} for engine, _, port in endpoints
        }, self.timeout_seconds)
        self.receivers_initialized = True

    def publish(self):
        harness = self.harness
        context = ThreadPoolExecutor(max_workers=1) if harness.is_driver else nullcontext()
        with context as executor:
            future = None
            if executor is not None:
                endpoints = publication_endpoints(harness)
                future = executor.submit(_request_endpoints, endpoints, "publication_update", {
                    engine: {"step_id": int(harness.megatron_engine.global_step)}
                    for engine, _, _ in endpoints
                }, self.timeout_seconds)
            harness._training_barrier()
            if not self.connected:
                # This is a new TCPStore, not torchrun's existing agent store.
                with _independent_rendezvous():
                    self.adapter.connect()
                self.connected = True
            event = self.adapter.send_weights()
            if event is not None:
                event.synchronize()
            # A pack-complete event alone is not end-to-end delivery.
            self.adapter.wait_send_complete()
            if future is not None:
                future.result()
            self.adapter.flush_profile_outputs()

    def close(self):
        if self.harness.is_driver and self.receivers_initialized:
            endpoints = publication_endpoints(self.harness)
            _request_endpoints(endpoints, "publication_close", {engine: {} for engine, _, _ in endpoints}, 60)
