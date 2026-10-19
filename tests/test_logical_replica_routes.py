"""CPU end-to-end checks of ownership, native descriptors and relay streams."""

from collections import defaultdict
from dataclasses import replace

import pytest
import torch

from shardstream import transport
from shardstream.metadata.parameter import ParameterReplicaMeta
from shardstream.plan import TransferPlan, TransferPlanBuilder
from test_dp_source_plan import parameter


def _local(operations, rank, sender):
    result = defaultdict(list)
    for op in operations:
        if (op.send_rank if sender else op.recv_rank) == rank:
            result[op.recv_rank if sender else op.send_rank].append(op)
    return TransferPlan(operations=dict(result))


def _signature(op):
    return (op.send_rank, op.recv_rank, op.send_offset, op.recv_offset,
            op.overlap_shape, op.replica_targets, op.replica_route_id)


@pytest.mark.parametrize("train_tp,train_copies,dp,cp,infer_tp,infer_copies,engines,axis", [
    (1, 4, 4, 1, 1, 4, 1, 0),  # one instance containing four DP replicas
    (2, 4, 2, 2, 2, 2, 2, 0),  # trainer DP x CP, rollout DP and multiple engines
    (2, 4, 2, 2, 4, 2, 2, 1),  # row/column strides and TP resharding
    (1, 4, 1, 1, 1, 4, 2, 0),  # model-internal replication without DP identities
    (4, 1, 1, 1, 2, 2, 1, 1),  # training DP1, rollout DP2
    (1, 1, 1, 1, 1, 4, 2, 0),  # a single source feeding every replica
])
def test_complete_updates_follow_the_prepared_replica_chains(
    monkeypatch, train_tp, train_copies, dp, cp, infer_tp, infer_copies, engines, axis,
):
    monkeypatch.setattr(transport, "_ensure_cuda_tensor", lambda *_: None)
    shape = (64, 32)
    trains = [parameter(name, shape, train_tp, train_copies, dp=dp, cp=cp, axis=axis)
              for name in ("a", "b", "c")]
    infers = [parameter(p.name, shape, infer_tp, infer_copies, dp=infer_copies, axis=axis)
              for p in trains]
    iw = infer_tp * infer_copies * engines
    tw = train_tp * train_copies
    builder = TransferPlanBuilder(iw, tw, engines, group_replicated_inference=True,
                                  dp_source_policy="stream", replica_routing="logical")
    ops = builder.build_weights_mapping_operations(infers, trains)
    for rank in range(iw + tw):
        local = builder.build_weights_mapping_operations(infers, trains, rank)
        assert [_signature(o) for o in local] == [_signature(o) for o in ops
                                                if rank in (o.send_rank, o.recv_rank)]
    # Engines may span nodes; DP replicas of one engine are deliberately spread.
    node_ids = tuple([r % 2 for r in range(iw)] + [10 + r // max(1, tw // 2) for r in range(tw)])
    engine_nodes = tuple(node_ids[e * infer_tp * infer_copies] for e in range(engines))
    values = {}
    for rank in range(iw + tw):
        metas = infers if rank < iw else trains
        shard_rank = rank % (infer_tp * infer_copies) if rank < iw else rank - iw
        values[rank] = {p.name: torch.zeros(next(s.shape for s in p.shards if s.global_rank == shard_rank))
                        for p in metas}
    options = dict(world_size=iw + tw, chunk_bytes=128, allow_staging=False,
                   infer_instance_world_size=infer_tp * infer_copies,
                   num_infer_engines=engines, ring_broadcast=True, ring_swizzle=True,
                   rollout_node_ids=engine_nodes, rank_node_ids=node_ids)
    sends = {r: transport._build_send_batch(values[r], _local(ops, r, True), r, **options)
             for r in range(iw, iw + tw)}
    recvs = {r: transport._build_recv_batch(values[r], _local(ops, r, False), r, **options)
             for r in range(iw)}
    expected_bytes = len(trains) * shape[0] * shape[1] * 4
    assert sum(sum(b.lengths) for b in sends.values()) == expected_bytes
    assert builder.logical_replica_stats["unique_source_bytes"] == expected_bytes

    for version in (1, 2):
        truths = {p.name: (torch.arange(shape[0] * shape[1]).reshape(shape).float()
                            + 10000 * version + index)
                  for index, p in enumerate(trains)}
        for rank in range(iw, iw + tw):
            for p in trains:
                s = next(s for s in p.shards if s.global_rank == rank - iw)
                values[rank][p.name].copy_(truths[p.name][tuple(
                    slice(a, a + n) for a, n in zip(s.global_offset, s.shape))])
        for rank in range(iw):
            for value in values[rank].values():
                value.fill_(-1)
        messages = defaultdict(list)
        for rank, batch in sends.items():
            for index, tensor in enumerate(batch.tensors):
                messages[(rank, batch.peers[index], batch.ring_ids[index])].append((
                    batch.lengths[index], tensor.reshape(-1)[
                        batch.tensor_offsets[index] // 4:
                        (batch.tensor_offsets[index] + batch.lengths[index]) // 4].clone()))
        pending = {}
        for rank, batch in recvs.items():
            for index in range(len(batch.tensors)):
                key = (batch.peers[index], rank, batch.ring_ids[index])
                pending.setdefault(key, []).append(index)
        while pending:
            advanced = False
            for key, indices in list(pending.items()):
                if key not in messages:
                    continue
                rank = key[1]
                batch = recvs[rank]
                payloads = messages[key]
                assert [batch.lengths[i] for i in indices] == [size for size, _ in payloads]
                for index, (_, payload) in zip(indices, payloads):
                    tensor = batch.tensors[index]
                    # Copy through the descriptor's row stride (including TP column views).
                    flat = tensor.as_strided((tensor.numel(),), (1,)) if tensor.is_contiguous() else None
                    if flat is not None:
                        offset = batch.tensor_offsets[index] // 4
                        flat[offset:offset + payload.numel()].copy_(payload)
                    else:
                        row_elements = batch.tensor_row_bytes[index] // 4
                        flat_offset = batch.tensor_offsets[index] // 4
                        for j, value in enumerate(payload):
                            row, col = divmod(flat_offset + j, row_elements)
                            tensor[row, col] = value
                following = batch.forward_peers[indices[0]]
                if following >= 0:
                    messages[(rank, following, key[2])] = payloads
                del pending[key]
                advanced = True
            assert advanced, "Replica descriptors contain a deadlock or unmatched stream"
        for rank in range(iw):
            for p in infers:
                s = next(s for s in p.shards if s.global_rank == rank % (infer_tp * infer_copies))
                assert torch.equal(values[rank][p.name], truths[p.name][tuple(
                    slice(a, a + n) for a, n in zip(s.global_offset, s.shape))])


def test_different_source_and_destination_replica_geometries_share_intervals():
    training = parameter("w", (64, 32), 2, 1, dp=1)
    extra = parameter("w", (64, 32), 4, 1, dp=1)
    shards = training.shards + [replace(s, global_rank=s.global_rank + 2) for s in extra.shards]
    training = replace(training, shards=shards, replicas=[
        ParameterReplicaMeta(training.shards), ParameterReplicaMeta(shards[2:])])
    inference = parameter("w", (64, 32), 1, 2, dp=2)
    builder = TransferPlanBuilder(2, 6, group_replicated_inference=True,
                                  dp_source_policy="stream", replica_routing="logical")
    ops = builder.build_weights_mapping_operations([inference], [training])
    masks = [torch.zeros((64, 32), dtype=torch.int32) for _ in range(2)]
    for op in ops:
        masks[op.recv_rank][op.inf_slices] += 1
        assert op.replica_targets == (0, 1)
    assert all(torch.all(mask == 1) for mask in masks)
    assert builder.logical_replica_stats["unique_source_bytes"] == 64 * 32 * 4


def test_missing_training_coverage_is_rejected():
    train = parameter("w", (64, 32), 2, 1, dp=1)
    train = replace(train, shards=train.shards[:1])
    infer = parameter("w", (64, 32), 1, 1, dp=1)
    builder = TransferPlanBuilder(1, 2, group_replicated_inference=True,
                                  dp_source_policy="stream", replica_routing="logical")
    with pytest.raises(ValueError, match="Uncovered training region"):
        builder.build_weights_mapping_operations([infer], [train])
