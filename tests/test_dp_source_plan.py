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
@pytest.mark.parametrize("policy", ["block", "stream"])
def test_tp4_cp2_dp2_exact_coverage_and_local_agreement(infer_tp, engines, axis, ring, policy):
    shape = (128, 64)
    training = parameter("weight", shape, tp=4, copies=4, dp=2, cp=2, axis=axis)
    inference = parameter("weight", shape, tp=infer_tp, copies=1, dp=1, axis=axis)
    builder = TransferPlanBuilder(infer_world_size=infer_tp * engines,
                                 train_world_size=16, num_infer_engines=engines,
                                 group_replicated_inference=ring,
                                 dp_source_policy=policy, dp_source_block_bytes=256)
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


@pytest.mark.parametrize("policy", ["block", "stream"])
def test_dp1_retains_existing_plan_and_metadata_copy_preserves_axes(policy):
    training = parameter("weight", (32, 16), 4, 2, dp=1, cp=2)
    inference = parameter("weight", (32, 16), 8, 1, dp=1)
    options = dict(infer_world_size=16, train_world_size=8, num_infer_engines=2,
                   group_replicated_inference=True)
    old = TransferPlanBuilder(**options, dp_source_policy="replica")
    new = TransferPlanBuilder(**options, dp_source_policy=policy)
    assert [signature(o) for o in old.build_weights_mapping_operations([inference], [training])] == [signature(o) for o in new.build_weights_mapping_operations([inference], [training])]
    copied = training.fast_copy_with_engine_rank(3)
    assert [(s.dp_rank, s.dp_size, s.cp_rank, s.cp_size) for s in copied.shards] == [(s.dp_rank, s.dp_size, s.cp_rank, s.cp_size) for s in training.shards]


@pytest.mark.parametrize("policy,size", [("bad", 1024), ("block", 0)])
def test_invalid_configuration(policy, size):
    with pytest.raises(ValueError):
        TransferPlanBuilder(1, 2, dp_source_policy=policy, dp_source_block_bytes=size)


@pytest.mark.parametrize("policy", ["block", "stream"])
def test_different_replica_geometry_retains_replica_policy(policy):
    training = parameter("weight", (128, 64), 2, 2)
    for index, shard in enumerate(training.replicas[1].shards):
        shard.shape = (128, 32)
        shard.global_offset = (0, index * 32)
        shard.sharding_dim = 1
    inference = parameter("weight", (128, 64), 2, 1, dp=1)
    options = dict(infer_world_size=4, train_world_size=4, num_infer_engines=2)
    original = TransferPlanBuilder(**options, dp_source_policy="replica")
    block = TransferPlanBuilder(**options, dp_source_policy=policy)
    assert [signature(o) for o in original.build_weights_mapping_operations([inference], [training])] == [signature(o) for o in block.build_weights_mapping_operations([inference], [training])]


def test_partial_final_block_preserves_every_row():
    training = parameter("weight", (10, 64), 1, 2)
    inference = parameter("weight", (10, 64), 1, 1, dp=1)
    builder = TransferPlanBuilder(1, 2, dp_source_policy="block", dp_source_block_bytes=3 * 64 * 4)
    ops = builder.build_weights_mapping_operations([inference], [training])
    assert [op.overlap_shape for op in ops] == [(3, 64), (3, 64), (3, 64), (1, 64)]
    assert [op.recv_offset for op in ops] == [(0, 0), (3, 0), (6, 0), (9, 0)]


def test_stream_only_splits_parameter_containing_cumulative_midpoint():
    training = [parameter(name, (rows, 8), 1, 2)
                for name, rows in [("a", 2), ("b", 6), ("c", 2)]]
    inference = [parameter(p.name, p.global_shape, 1, 1, dp=1) for p in training]
    builder = TransferPlanBuilder(2, 2, 2, group_replicated_inference=True,
                                 dp_source_policy="stream", dp_source_block_bytes=1)
    ops = builder.build_weights_mapping_operations(inference, training)
    assert [(o.send_shard_meta.name, o.send_shard_meta.dp_rank, o.send_offset[0], o.overlap_shape[0])
            for o in ops if o.recv_rank == 0] == [
                ("a", 0, 0, 2), ("b", 0, 0, 3), ("b", 1, 3, 3), ("c", 1, 0, 2)]
    assert builder.dp_stream_stats == dict(groups=1, shard_regions=3, fragments=4,
                                         split_regions=1, bytes_by_rank={0:160, 1:160},
                                         bytes_by_dp={0:160, 1:160})
    shuffled = builder.build_weights_mapping_operations(inference[::-1], training[::-1])
    assert list(map(signature, ops)) == list(map(signature, shuffled))


def test_stream_partitions_each_tp_rank_and_preserves_parameter_edges():
    # Each DP spans two hypothetical nodes (TP0/1 and TP2/3). Corresponding
    # ranks split their own stream, rather than assigning whole model halves.
    training = [parameter(name, (16, 16), 4, 2) for name in ("a", "b")]
    inference = [parameter(p.name, p.global_shape, 4, 1, dp=1) for p in training]
    builder = TransferPlanBuilder(4, 8, dp_source_policy="stream")
    ops = builder.build_weights_mapping_operations(inference, training)
    assert len(ops) == 8
    assert builder.dp_stream_stats['groups'] == 4
    assert builder.dp_stream_stats['split_regions'] == 0
    assert set(builder.dp_stream_stats['bytes_by_rank'].values()) == {4 * 16 * 4}
    assert {o.send_rank for o in ops if o.send_shard_meta.name == "a"} == set(range(4, 8))
    assert {o.send_rank for o in ops if o.send_shard_meta.name == "b"} == set(range(8, 12))


def test_stream_odd_rows_and_tiny_parameters_have_no_gaps():
    training = [parameter("a", (1, 8), 1, 2), parameter("b", (4, 8), 1, 2)]
    inference = [parameter(p.name, p.global_shape, 1, 1, dp=1) for p in training]
    builder = TransferPlanBuilder(1, 2, dp_source_policy="stream")
    ops = builder.build_weights_mapping_operations(inference, training)
    assert [(o.send_shard_meta.name, o.send_shard_meta.dp_rank, o.send_offset[0], o.overlap_shape[0]) for o in ops] == [
        ("a", 0, 0, 1), ("b", 0, 0, 1), ("b", 1, 1, 3)]
    assert builder.dp_stream_stats['bytes_by_dp'] == {0:64, 1:96}
