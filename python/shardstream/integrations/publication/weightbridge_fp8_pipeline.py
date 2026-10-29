"""Round-ready preparation on native WeightBridge's double-buffered send loop.

Persistent model staging remains allocated. Only its production is pipelined;
native wire-buffer lifetime gates and background RDMA are unchanged.
"""
import hashlib
import json
import time

import torch
import torch.distributed as dist


def preparation_schedule(global_rounds, names, physical_name):
    names = set(names)
    first = {}
    for index, sources in enumerate(global_rounds):
        for source in sources:
            name = physical_name(source)
            if name.endswith('_scale_inv'):
                name = name[:-len('_scale_inv')]
            if name in names:
                first.setdefault(name, index)
    if names != set(first):
        raise ValueError(f'Quantized tensors absent from native rounds: {sorted(names - set(first))[:8]}')
    return [sorted(name for name, first_round in first.items() if first_round == index)
            for index in range(len(global_rounds))]


def overlap_pairs(preparation, wire):
    """Host preparation intervals intersecting older native RDMA envelopes."""
    result = []
    for current, start, end in preparation:
        for previous, (sent, done) in wire.items():
            overlap = min(end, done) - max(start, sent)
            if previous < current and overlap > 0:
                result.append(dict(preparing_round=current, transmitting_round=previous,
                                   overlap_ms=overlap / 1e6))
    return result


class RoundPreparation:
    def __init__(self, quantizer, sender):
        from wbridge.utils.data import parse_logical_tensor_name
        from shardstream.integrations.publication.weightbridge_fp8_model import is_quantized
        if sender.sender_staging or getattr(sender, '_relay_enabled', False) or sender._NUM_BUF != 2:
            raise ValueError('FP8 pipeline requires native direct GPU RDMA with two wire buffers')
        self.quantizer, self.sender = quantizer, sender
        def physical(name):
            parsed = parse_logical_tensor_name(name)
            return parsed[0] if parsed is not None else name
        self.schedule = preparation_schedule(sender.router.global_rounds,
            [name for name in quantizer.source if is_quantized(name)], physical)
        # Use the global round plan, including rounds with no local wire data.
        # TP partners thus issue identical collectives even when their local
        # sender loops skip different rounds. All tensors are prepared once,
        # retaining eager's work/traffic rather than silently dropping replicas.
        gather_order = [[name for name in names if name in quantizer.gathers] for names in self.schedule]
        peers = [None] * quantizer.world
        dist.all_gather_object(peers, gather_order, group=quantizer.group)
        if any(peer != gather_order for peer in peers):
            raise ValueError('TP ranks disagree on native-round gather order')
        self.epoch = None
        self.events = []
        original_trace = sender._trace_state
        def trace(event, **fields):
            if fields.get('epoch') == self.epoch:
                if event == 'send_pack':
                    self.prepare_through(fields['round'])
                elif event == 'send_enqueued':
                    # Ranks without trailing local traffic must still join TP
                    # gathers needed by their partners before leaving send().
                    self.prepare_through(len(self.schedule) - 1)
                if event in ('send_pack', 'send_round_queued', 'sender_peer_bulk_submit', 'sender_round_done'):
                    self.events.append((event, fields['round'], time.perf_counter_ns()))
            return original_trace(event, **fields)
        sender._trace_state = trace
        plan = [sorted(names) for names in sender.router.global_rounds]
        print('WB_FP8_PIPELINE_PLAN ' + json.dumps(dict(rank=sender.rank,
            native_rounds=len(plan), native_wire_buffers=sender._NUM_BUF,
            quantized_tensors_by_round=list(map(len, self.schedule)),
            gathered_tensors_by_round=list(map(len, gather_order)),
            global_plan_sha256=hashlib.sha256(json.dumps(plan).encode()).hexdigest(),
            source_staging='persistent_full_local_payload',
            scope='prepare next native round while prior wire buffers are in flight')), flush=True)

    @torch.no_grad()
    def begin(self, step):
        self.step, self.epoch = step, self.sender._epoch
        self.next_round, self.preparation, self.events, self.gpu_intervals = 0, [], [], []
        started = time.perf_counter_ns()
        self.quantizer.extract.run()
        self.extract_enqueue_ms = (time.perf_counter_ns() - started) / 1e6

    @torch.no_grad()
    def prepare_through(self, target):
        for index in range(self.next_round, target + 1):
            names = self.schedule[index]
            if names:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter_ns()
                start.record()
                self.quantizer.prepare_names(names)
                end.record()
                self.preparation.append((index, begin, time.perf_counter_ns()))
                self.gpu_intervals.append((start, end))
            self.next_round = index + 1

    def finish_enqueue(self):
        if self.next_round != len(self.schedule):
            raise RuntimeError('Native sender did not reach the expected round hooks')
        # Pack-ready events still gate RDMA. This final stream drain also covers
        # trailing preparation on ranks with no local wire use in those rounds.
        torch.cuda.current_stream().synchronize()

    def report(self):
        starts, ends = {}, {}
        for event, index, timestamp in self.events:
            if event == 'sender_peer_bulk_submit':
                starts.setdefault(index, timestamp)
            elif event == 'sender_round_done':
                ends[index] = timestamp
        if set(starts) - set(ends):
            raise RuntimeError('Pipeline report requested before native RDMA completion')
        wire = {index: (start, ends[index]) for index, start in starts.items()}
        prepared = {index for index, _, _ in self.preparation}
        packed = [index for event, index, _ in self.events if event == 'send_pack']
        expected = [i for i, (_, overlaps) in enumerate(self.sender.router.local_rounds) if overlaps]
        if packed != expected or prepared != {i for i, names in enumerate(self.schedule) if names}:
            raise RuntimeError('Incomplete native round preparation/pack coverage')
        return dict(rank=self.sender.rank, step=self.step, preparation_mode='pipeline',
            full_tensor_quantizations=sum(map(len, self.schedule)),
            gathered_tensors=len(self.quantizer.gathers),
            logical_trainer_gather_egress_bytes=self.quantizer.extra_bytes,
            prepare_ms=sum(end - start for _, start, end in self.preparation) / 1e6 + self.extract_enqueue_ms,
            prepare_timing='sum_host_enqueue_intervals_not_eager_wall_time',
            prepare_cuda_stream_span_ms=sum(start.elapsed_time(end) for start, end in self.gpu_intervals),
            overlap_pairs=overlap_pairs(self.preparation, wire),
            overlap_scope='host preparation vs native RDMA submit-to-completion envelopes, not kernel utilization',
            preparation_intervals_ns=self.preparation, native_rdma_intervals_ns=wire,
            native_pack_rounds=packed)
