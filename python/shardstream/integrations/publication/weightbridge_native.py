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
    if RECEIVER_ROUND_CAP_BYTES != int(round_bytes):
        raise RuntimeError(f"WeightBridge imported with round cap {RECEIVER_ROUND_CAP_BYTES}; requested {round_bytes}")
    print("WB_ROUND_CAP " + json.dumps({"pid": os.getpid(), "requested_bytes": int(round_bytes),
                                      "effective_bytes": RECEIVER_ROUND_CAP_BYTES}), flush=True)


def _parameter_views(model):
    return {name: param.detach() for name, param in model.named_parameters()}


@contextmanager
def _model_probe_guard(model, role, rank):
    """Check specgen restoration against the framework-loaded checkpoint.

    WeightBridge restores and verifies through its inferred LoadSpec. Keep an
    independent reference from before probing, outside publication timing, so
    a wrong mapping cannot become the reference for subsequent update checks.
    """
    expected = {name: value.clone() for name, value in _parameter_views(model).items()}
    yield
    actual = _parameter_views(model)
    bad = sorted(set(actual) ^ set(expected))
    for name in expected.keys() & actual.keys():
        a, b = actual[name], expected[name]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
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


def _storage_parameter_views(model):
    """Bind native storage once when strided parameters alias a full owner.

    A reshape of a strided view can allocate a detached copy. Prefer an
    existing contiguous parameter covering its complete storage. Black-box
    probing still discovers every correspondence; no mapping is supplied.
    """
    values = _parameter_views(model)
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


def _normalize_probe_views(factories, shapes, model, loader):
    """Expose fused HF tensors as matrix views to the native black-box probe.

    Framework loaders continue receiving the original checkpoint shapes. The
    probe alone sees row-major views; worker vector parameters get one leading
    singleton axis. No values, storage pointers or inferred mappings are supplied.
    """
    import math
    fused = {name: shape for name, shape in shapes.items() if len(shape) > 2}
    if not fused:
        return factories, shapes, lambda: _storage_parameter_views(model), loader
    normalized = dict(factories)
    normalized_shapes = dict(shapes)
    for name, shape in fused.items():
        matrix_shape = (math.prod(shape[:-1]), shape[-1])
        normalized_shapes[name] = matrix_shape
        def fetch(factory=factories[name], matrix_shape=matrix_shape):
            return factory().view(matrix_shape)
        normalized[name] = fetch

    def load(probes):
        original = dict(probes)
        for name, shape in fused.items():
            if name not in probes:
                continue
            def fetch(factory=probes[name], shape=shape):
                return factory().view(shape)
            original[name] = fetch
        loader(original)

    def worker_views():
        return {name: value.unsqueeze(0) if value.ndim == 1 else value
                for name, value in _storage_parameter_views(model).items()}
    print("WB_PROBE_MATRIX_VIEWS " + json.dumps({
        "pid": os.getpid(), "fused_sources": len(fused),
        "checkpoint_values_unchanged": True,
    }), flush=True)
    return normalized, normalized_shapes, worker_views, load


def _context(model_path, model, rank, loader):
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
    storage_views = _storage_parameter_views(model)
    print("WB_STORAGE_VIEWS " + json.dumps({"pid": os.getpid(), "rank": rank,
        "model_parameters": len(_parameter_views(model)), "storage_views": len(storage_views),
        "strided_aliases_covered_by_owner": len(_parameter_views(model)) - len(storage_views)}), flush=True)
    factories, shapes, worker_views, loader = _normalize_probe_views(
        factories, shapes, model, loader)
    return AdapterContext(factories, shapes, worker_views, loader, rank)


def _load_vllm_weights(model_runner, factories):
    """Probe the complete native loader, including derived attention weights."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    model = model_runner.model
    with torch.no_grad():
        model.load_weights((name, factory()) for name, factory in factories.items())
        process_weights_after_loading(
            model, model_runner.model_config, next(model.parameters()).device)


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

        model = self.model_runner.model
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
                harness.inference_config["model_path"], harness.mcore_model, harness.rank, load), args)
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
