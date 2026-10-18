"""Coordinate-sensitive simulation of replicated training source selection."""

from collections import Counter

import pytest
import torch

from shardstream.metadata.parameter import ParameterMeta, ParameterReplicaMeta, ParameterShardMeta
from shardstream.metadata.sharding import ShardingType
from shardstream.plan import TransferPlanBuilder


def parameter(name, shape, tp, copies, dp=2, cp=1, axis=0):
    replicas = []
    for copy in range(copies):
        shards = []
        for t in range(tp):
            local = list(shape)
            local[axis] //= tp
            offset = [0] * len(shape)
            offset[axis] = t * local[axis]
            shards.append(ParameterShardMeta(
                name=name, tp_rank=t, attn_tp_rank=t, pp_rank=0,
                ep_rank=0, ep_tp_rank=t, global_rank=copy * tp + t,
                world_size=tp * copies, engine_rank=0, shape=tuple(local),
                numel=int(torch.tensor(local).prod()), dtype=torch.float32,
                global_offset=tuple(offset), num_shards=tp, sharding_dim=axis,
                sharding_type=ShardingType.TP_SHARDING, dp_size=dp,
                dp_rank=copy // cp, cp_rank=copy % cp, cp_size=cp,
            ))
        replicas.append(ParameterReplicaMeta(shards=shards))
    return ParameterMeta(name=name, global_shape=shape,
                         global_numel=int(torch.tensor(shape).prod()), dtype=torch.float32,
                         shards=[s for r in replicas for s in r.shards], replicas=replicas)


def signature(op):
    return (op.send_rank, op.recv_rank, op.send_offset, op.recv_offset,
            op.overlap_shape, op.send_shard_meta.name)


@pytest.mark.parametrize("infer_tp,engines", [(8, 2), (4, 4)])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("ring", [False, True])
def test_tp4_cp2_dp2_exact_coverage_and_local_agreement(infer_tp, engines, axis, ring):
    shape = (128, 64)
    training = parameter("weight", shape, tp=4, copies=4, dp=2, cp=2, axis=axis)
    inference = parameter("weight", shape, tp=infer_tp, copies=1, dp=1, axis=axis)
    builder = TransferPlanBuilder(infer_world_size=infer_tp * engines,
                                 train_world_size=16, num_infer_engines=engines,
                                 group_replicated_inference=ring,
                                 dp_source_policy="block", dp_source_block_bytes=256)
    ops = builder.build_weights_mapping_operations([inference], [training])
    assert [signature(o) for o in ops] == [signature(o) for o in builder.build_weights_mapping_operations([inference], [training])]
    truth = torch.arange(1, 1 + shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    outputs, masks = {}, {}
    for engine in range(engines):
        for shard in inference.shards:
            rank = engine * infer_tp + shard.global_rank
            outputs[rank] = torch.zeros(shard.shape)
            masks[rank] = torch.zeros(shard.shape, dtype=torch.int32)
    for op in ops:
        shard = op.send_shard_meta
        source = truth[tuple(slice(a, a + b) for a, b in zip(shard.global_offset, shard.shape))]
        outputs[op.recv_rank][op.inf_slices] = source[op.train_slices]
        masks[op.recv_rank][op.inf_slices] += 1
    for rank, output in outputs.items():
        shard = inference.shards[rank % infer_tp]
        expected = truth[tuple(slice(a, a + b) for a, b in zip(shard.global_offset, shard.shape))]
        assert torch.equal(output, expected)
        assert torch.all(masks[rank] == 1)
    assert {op.send_shard_meta.dp_rank for op in ops} == {0, 1}
    assert len({op.send_rank for op in ops}) == 16
    for rank in range(infer_tp * engines + 16):
        pruned = builder.build_weights_mapping_operations([inference], [training], global_transfer_rank=rank)
        assert [signature(o) for o in pruned] == [signature(o) for o in ops if rank in (o.send_rank, o.recv_rank)]
    if ring:
        for source in {op.send_rank for op in ops}:
            for peer in range(infer_tp):
                streams = [[(o.send_offset, o.recv_offset, o.overlap_shape) for o in ops
                            if o.send_rank == source and o.recv_rank == engine * infer_tp + peer]
                           for engine in range(engines)]
                assert all(s == streams[0] for s in streams)


def test_expert_dp2_and_cross_parameter_load_balance():
    training = [parameter(f"expert.{i}", (128, 64), 1, 2) for i in range(3)]
    inference = [parameter(p.name, (128, 64), 1, 1, dp=1) for p in training]
    builder = TransferPlanBuilder(2, 2, 2, group_replicated_inference=True,
                                 dp_source_policy="block", dp_source_block_bytes=2048)
    ops = builder.build_weights_mapping_operations(inference, training)
    loads = Counter()
    for op in ops:
        if op.recv_rank == 0:
            loads[op.send_rank] += int(torch.tensor(op.overlap_shape).prod()) * 4
    assert loads[2] == loads[3]


def test_dp1_retains_existing_plan_and_metadata_copy_preserves_axes():
    training = parameter("weight", (32, 16), 4, 2, dp=1, cp=2)
    inference = parameter("weight", (32, 16), 8, 1, dp=1)
    options = dict(infer_world_size=16, train_world_size=8, num_infer_engines=2,
                   group_replicated_inference=True)
    old = TransferPlanBuilder(**options, dp_source_policy="replica")
    new = TransferPlanBuilder(**options, dp_source_policy="block")
    assert [signature(o) for o in old.build_weights_mapping_operations([inference], [training])] == [signature(o) for o in new.build_weights_mapping_operations([inference], [training])]
    copied = training.fast_copy_with_engine_rank(3)
    assert [(s.dp_rank, s.dp_size, s.cp_rank, s.cp_size) for s in copied.shards] == [(s.dp_rank, s.dp_size, s.cp_rank, s.cp_size) for s in training.shards]


@pytest.mark.parametrize("policy,size", [("bad", 1024), ("block", 0)])
def test_invalid_configuration(policy, size):
    with pytest.raises(ValueError):
        TransferPlanBuilder(1, 2, dp_source_policy=policy, dp_source_block_bytes=size)
