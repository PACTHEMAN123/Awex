def get_sharding_strategy_builder(engine_name: str):
    if engine_name == "sglang":
        raise NotImplementedError(
            "SGlang engine is outside the migrated experiment matrix"
        )

    if engine_name == "vllm":
        from shardstream.integrations.sharding.vllm import get_vllm_sharding_strategy

        return get_vllm_sharding_strategy
    if engine_name == "mcore":
        from shardstream.integrations.sharding.megatron import (
            get_mcore_sharding_strategy,
        )

        return get_mcore_sharding_strategy
    raise ValueError(f"Unknown engine_name {engine_name}")


def get_rank_info_extractor(engine_name: str):
    if engine_name == "sglang":
        raise NotImplementedError(
            "SGlang engine is outside the migrated experiment matrix"
        )

    if engine_name == "vllm":
        from shardstream.integrations.sharding.vllm import get_vllm_rank_info

        return get_vllm_rank_info
    if engine_name == "mcore":
        from shardstream.integrations.sharding.megatron import get_mcore_rank_info

        return get_mcore_rank_info
    raise ValueError(f"Unknown engine_name {engine_name}")
