"""Reconstruct GQA KV shapes and offsets while retaining every physical receiver."""

from types import SimpleNamespace

import pytest
import torch

from awex.meta.infer_meta_resolver import InferParamMetaResolver
from awex.models.canonical_vllm import Qwen3ShardingStrategy
from awex.sharding.rank_info import RankInfo


@pytest.mark.parametrize(
    "tp,kv_heads,dp_attention",
    [
        (8, 4, False),
        (8, 1, False),
        (8, 8, False),
        (4, 4, False),
        (2, 4, False),
        (8, 4, True),
    ],
)
def test_qwen_gqa_replicas_reconstruct_one_logical_tensor(tp, kv_heads, dp_attention):
    world_size = 2 * tp
    config = SimpleNamespace(num_key_value_heads=kv_heads)
    rank0 = SimpleNamespace(tp_size=tp, attn_tp_size=tp)
    strategy = Qwen3ShardingStrategy(
        engine_name="vllm",
        enable_dp_attention=dp_attention,
        enable_dp_lm_head=False,
        moe_dense_tp_size=tp,
        tp_size=tp,
        ep_size=1,
        ep_tp_size=tp,
        rank_info=rank0,
        hf_config=config,
    )
    q_name = "model.layers.0.self_attn.q_proj.weight"
    k_name = "model.layers.0.self_attn.k_proj.weight"
    v_name = "model.layers.0.self_attn.v_proj.weight"
    input_dim, head_dim = 3, 2
    global_k = torch.arange(kv_heads * head_dim * input_dim).reshape(
        kv_heads * head_dim, input_dim
    )
    global_q = torch.arange(tp * head_dim * input_dim).reshape(tp * head_dim, input_dim)
    logical_shards = min(tp, kv_heads)
    replica_factor = max(1, tp // kv_heads)
    tensors = {}
    raw = []
    for rank in range(world_size):
        tp_rank = rank % tp
        kv_rank = tp_rank // replica_factor
        local_k = global_k.chunk(logical_shards)[kv_rank]
        local_q = global_q.chunk(tp)[tp_rank]
        params = []
        for name, tensor in [
            (q_name, local_q),
            (k_name, local_k),
            (v_name, local_k + 100),
        ]:
            tensors[rank, name] = tensor
            params.append(
                {
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "numel": tensor.numel(),
                    "dtype": torch.int64,
                }
            )
        info = RankInfo(
            global_rank=rank,
            world_size=world_size,
            tp_rank=tp_rank,
            tp_size=tp,
            pp_rank=0,
            pp_size=1,
            dp_size=2,
            dp_rank=rank // tp,
            ep_rank=0,
            ep_size=1,
            ep_tp_rank=tp_rank,
            ep_tp_size=tp,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp,
            attn_dp_rank=rank // tp,
            local_rank=tp_rank,
            engine_rank=rank // tp,
            is_infer=True,
        )
        raw.append({"rank_info": info, "params_meta": params})
    resolver = object.__new__(InferParamMetaResolver)
    resolver._sharding_strategy = strategy
    resolver._params_raw_meta = raw
    metadata = {param.name: param for param in resolver._build_params_meta()}
    for name, expected in [
        (q_name, global_q),
        (k_name, global_k),
        (v_name, global_k + 100),
    ]:
        meta = metadata[name]
        assert meta.global_shape == tuple(expected.shape)
        assert meta.global_numel == expected.numel()
        assert len(meta.replicas) == (2 if name == q_name else 2 * replica_factor)
        assert len(meta.shards) == world_size
        assert {shard.global_rank for shard in meta.shards} == set(range(world_size))
        for replica in meta.replicas:
            reconstructed = torch.empty_like(expected)
            for shard in replica.shards:
                start = shard.global_offset[0]
                reconstructed[start : start + shard.shape[0]] = tensors[
                    shard.global_rank, name
                ]
                assert shard.tp_rank == shard.global_rank % tp
                assert shard.engine_rank == shard.global_rank // tp
            assert torch.equal(reconstructed, expected)
