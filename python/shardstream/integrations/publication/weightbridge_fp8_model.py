"""Experimental real-model gather/quantize frontend over native WeightBridge.

Qwen3-MoE, block-aligned TP partitions and Triton FP8 inference layouts only.
Native specgen discovers the real Megatron and vLLM correspondences. Source
staging is eager per publication; no overlap with native RDMA rounds is claimed.
"""
import json
import os
import time

import torch
import torch.distributed as dist

from shardstream.integrations.publication.registry import (
    publication_endpoints, register_publication_mechanism, register_vllm_publication_receiver,
)
from shardstream.integrations.publication.weightbridge_native import (
    ProbeSource, WeightBridgePublication, WeightBridgeReceiver, _configure, _context,
    _model_probe_guard,
)


def is_quantized(name):
    return name.endswith(('_proj.weight',)) and ('.self_attn.' in name or '.mlp.experts.' in name)


def quantize(tensor):
    from verl_fp8_kernel import scaled_fp8_blockwise_triton
    return scaled_fp8_blockwise_triton(tensor.to(device='cuda', dtype=torch.bfloat16), (128, 128))


def quantized_catalog(factories, shapes):
    result, dimensions = dict(factories), dict(shapes)
    for name, shape in shapes.items():
        if not is_quantized(name):
            continue
        if len(shape) != 2 or any(n % 128 for n in shape):
            raise ValueError(f'Unsupported FP8 physical tensor: {name}: {shape}')
        factory = factories[name]
        result[name] = lambda factory=factory: quantize(factory())[0].view(torch.uint8)
        result[name + '_scale_inv'] = lambda factory=factory: quantize(factory())[1]
        dimensions[name + '_scale_inv'] = tuple(n // 128 for n in shape)
    return result, dimensions


def full_box(shape):
    return [(0, n, n) for n in shape]


def canonical_boxes(shape, shards):
    from wbridge.utils.data import _check_shard_compatibility
    result = []
    for shard in shards:
        aligned = _check_shard_compatibility(full_box(shape), shard)
        if aligned is None or len(aligned) != len(shape):
            raise ValueError(f'Cannot normalize source shard {shard} to {shape}')
        result.append([(lo, hi, width) for _, _, lo, hi, width in aligned])
    return result


class ModelGatherQuantizer:
    def __init__(self, ctx, load_spec, group):
        from wbridge.utils.data import ShardSpec, CopyPlan, shards_numel
        self.ctx, self.group, self.load_spec = ctx, group, load_spec
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.source_spec = load_spec.src_shard_spec
        self.wksd = ctx.wksd_factory()
        dtype_spec = {name: max((self.wksd[key].dtype for key in entry), key=lambda d: d.itemsize)
                      for name, entry in load_spec.entries.items()}
        self.source = self.source_spec.make_named_buffer(dtype_spec, 'cuda')
        self.extract = CopyPlan(load_spec.copy_fromto_pairs(
            self.source_spec, self.source, self.wksd, src_to_dst=False))
        peers = [None] * self.world
        dist.all_gather_object(peers, self.source_spec.entries, group=group)
        self.specs, self.payload, self.gathers = {}, {}, {}
        self.extra_bytes = 0
        for name, shards in self.source_spec:
            shape = ctx.hf_shapes[name]
            boxes = canonical_boxes(shape, shards)
            self.specs[name] = ShardSpec({name: boxes})
            if not is_quantized(name):
                self.payload[name] = self.source[name]
                continue
            if any(lo % 128 or hi % 128 or width % 128 for box in boxes for lo, hi, width in box):
                raise ValueError(f'Unaligned FP8 source: {name}: {boxes}')
            self.payload[name] = torch.empty_like(self.source[name], dtype=torch.uint8)
            scale_name = name + '_scale_inv'
            scale_boxes = [[(lo // 128, hi // 128, width // 128) for lo, hi, width in box] for box in boxes]
            self.specs[scale_name] = ShardSpec({scale_name: scale_boxes})
            self.payload[scale_name] = torch.empty(shards_numel(scale_boxes), dtype=torch.float32, device='cuda')
            if boxes == [full_box(shape)]:
                continue
            rank_boxes = [canonical_boxes(shape, p.get(name, [])) for p in peers]
            if any(not b for b in rank_boxes):
                raise ValueError(f'Full tensor cannot be gathered within its native TP group: {name}')
            counts = [shards_numel(b) for b in rank_boxes]
            if len(set(counts)) != 1 or sum(counts) != shape[0] * shape[1]:
                raise ValueError(f'Only equal-sized nonreplicated TP fragments supported: {name}')
            # Reject overlap rather than silently reconstruct from duplicate/stale fragments.
            for i, boxes_a in enumerate(rank_boxes):
                for boxes_b in rank_boxes[i + 1:]:
                    for a in boxes_a:
                        for b in boxes_b:
                            if all(max(x[0], y[0]) < min(x[1], y[1]) for x, y in zip(a, b)):
                                raise ValueError(f'Overlapping gather fragments: {name}')
            gathered = torch.empty(sum(counts), dtype=dtype_spec[name], device='cuda')
            full = torch.empty(shape, dtype=dtype_spec[name], device='cuda')
            full_spec = ShardSpec({name: [full_box(shape)]})
            pieces = {r: ShardSpec({name: b}) for r, b in enumerate(rank_boxes)}
            tensors = {r: {name: gathered[r * counts[0]:(r + 1) * counts[0]]} for r in pieces}
            reconstruct = CopyPlan(full_spec({name: full.flatten()}).setitem_named_pairs(pieces, tensors))
            self.gathers[name] = gathered, full, reconstruct
            self.extra_bytes += self.source[name].numel() * self.source[name].element_size() * (self.world - 1)
        # All partial tensors must issue the same collectives in each TP group.
        names = [None] * self.world
        dist.all_gather_object(names, sorted(self.gathers), group=group)
        if any(n != names[0] for n in names):
            raise ValueError('Native TP ranks disagree on full-tensor gather order')
        self.factories, self.shapes = quantized_catalog(ctx.hf_weights, ctx.hf_shapes)
        self.views = {name: value.view(1, -1) for name, value in self.payload.items()}

    def load_payload(self, factories):
        from wbridge.utils.data import BoundShardSpec, ShardSpec
        for name, target in self.payload.items():
            source = factories[name]().to(device=target.device, dtype=target.dtype).contiguous()
            spec = ShardSpec({name: [full_box(self.shapes[name])]})
            BoundShardSpec.slice_copy(spec({name: source.flatten()}), self.specs[name]({name: target}))

    def context(self, rank):
        from wbridge.frontend.adapters import AdapterContext
        return AdapterContext(self.factories, self.shapes, lambda: self.views, self.load_payload, rank)

    @torch.no_grad()
    def prepare(self):
        from wbridge.utils.data import BoundShardSpec, ShardSpec
        started = time.perf_counter()
        self.extract.run()
        count = 0
        for name in sorted(self.source):
            if not is_quantized(name):
                continue
            shape = self.ctx.hf_shapes[name]
            if name in self.gathers:
                gathered, full, reconstruct = self.gathers[name]
                dist.all_gather_into_tensor(gathered, self.source[name], group=self.group)
                reconstruct.run()
            else:
                full = self.source[name].view(shape)
            q, scale = quantize(full)
            for key, tensor in [(name, q.view(torch.uint8)), (name + '_scale_inv', scale)]:
                spec = ShardSpec({key: [full_box(self.shapes[key])]})
                BoundShardSpec.slice_copy(spec({key: tensor.flatten()}), self.specs[key]({key: self.payload[key]}))
            count += 1
        torch.cuda.synchronize()
        return dict(full_tensor_quantizations=count, gathered_tensors=len(self.gathers),
                    logical_trainer_gather_egress_bytes=self.extra_bytes,
                    prepare_ms=(time.perf_counter() - started) * 1000)


@torch.no_grad()
def load_native_fp8(model_runner, factories):
    from verl.utils.vllm.vllm_quant_utils import (
        prepare_quanted_weights_for_loading, process_quanted_weights_after_loading,
    )
    model = model_runner.model
    state = prepare_quanted_weights_for_loading(model)
    for name, parameter in model.named_parameters():
        parameter._verl_qualname = name
        if hasattr(parameter, 'subclass_type'):
            parameter.orig_type = parameter.__class__
            parameter.__class__ = parameter.subclass_type
    def weights():
        for name, factory in factories.items():
            value = factory()
            if is_quantized(name):
                value = value.to(torch.uint8).view(torch.float8_e4m3fn)
            yield name, value
    try:
        model.load_weights(weights())
    finally:
        for parameter in model.parameters():
            if hasattr(parameter, 'orig_type'):
                parameter.__class__ = parameter.orig_type
                del parameter.orig_type
    process_quanted_weights_after_loading(model, state)


@register_vllm_publication_receiver('weightbridge_gather_fp8')
class WeightBridgeFP8Receiver(WeightBridgeReceiver):
    def initialize(self):
        _configure(self.config['round_bytes'])
        from wbridge.backend import coordinator
        from wbridge.backend.control_channel import coordinator_ipc, coordinator_tcp_port
        from wbridge.frontend.adapters import ReceiverAdapter
        model = self.model_runner.model
        if model.config.architectures != ['Qwen3MoeForCausalLM']:
            raise ValueError('Experimental WeightBridge FP8 currently supports Qwen3-MoE only')
        port = self.config['server_port']
        ipc = coordinator_ipc(port)
        if self.worker_rank == 0:
            self.coordinator = coordinator.spawn(ipc, coordinator_tcp_port(port))
        ctx = _context(self.config['model_path'], model, self.worker_rank,
                       lambda f: load_native_fp8(self.model_runner, f), source_transform=quantized_catalog)
        views = ctx.wksd_factory
        ctx.wksd_factory = lambda: {name: value.view(torch.uint8) if value.dtype == torch.float8_e4m3fn else value
                                   for name, value in views().items()}
        started = time.perf_counter()
        self.adapter = ReceiverAdapter(ctx, ipc, num_workers=self.config['world_per_engine'])
        torch.cuda.synchronize()
        return dict(worker_rank=self.worker_rank, specgen_ms=(time.perf_counter() - started) * 1000,
                    native_fp8_payload=True)


@register_publication_mechanism('weightbridge_gather_fp8')
class WeightBridgeFP8Publication(WeightBridgePublication):
    def initialize_training(self):
        _configure(self.round_bytes)
        from megatron.core import parallel_state
        from wbridge.utils.specgen import infer_load_spec, verify_load_spec
        from wbridge.frontend.adapters import SenderAdapter
        from wbridge.backend.sender import SenderArgs
        from wbridge.backend.control_channel import coordinator_tcp_port
        from shardstream._utils.common import get_free_port
        harness = self.harness
        bridge = harness.mcore_bridge
        def loader(factories):
            with torch.no_grad():
                bridge._model_bridge.load_weights_hf_to_megatron(
                    ProbeSource(bridge.hf_pretrained, factories), [harness.mcore_model])
        with _model_probe_guard(harness.mcore_model, 'training', harness.rank):
            ctx = _context(harness.inference_config['model_path'], harness.mcore_model,
                           harness.rank, loader, include_buffers=True)
            spec = infer_load_spec(ctx.hf_weights, ctx.hf_shapes, ctx.wksd_factory, ctx.load_weights)
            verify_load_spec(ctx.hf_weights, ctx.wksd_factory, spec)
        self.quantizer = ModelGatherQuantizer(ctx, spec, parallel_state.get_tensor_model_parallel_group())
        port = [get_free_port() if harness.is_driver else None]
        dist.broadcast_object_list(port, src=0)
        args = SenderArgs(harness.world_size,
            [f'tcp://{host}:{coordinator_tcp_port(p)}' for _, host, p in publication_endpoints(harness)],
            os.environ['MASTER_ADDR'], port[0], protocol='rdma')
        self.adapter = SenderAdapter(self.quantizer.context(harness.rank), args)
        # Wrap only the frontend callback. Native send, routing and RDMA stay unchanged.
        native_send = self.adapter.send_weights
        def prepare_and_send():
            metrics = self.quantizer.prepare()
            print('WB_FP8_PREPARE ' + json.dumps(dict(rank=harness.rank,
                  step=int(harness.megatron_engine.global_step), **metrics)), flush=True)
            return native_send()
        self.adapter.send_weights = prepare_and_send
        torch.cuda.synchronize()
