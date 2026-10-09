# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from enum import Enum

from shardstream import logging
from shardstream.metadata.rank import RankInfo

logger = logging.getLogger(__name__)


class ShardingType(Enum):
    """
    Enum representing the type of sharding applied to a parameter.
    """

    NO_SHARDING = "NO_SHARDING"
    TP_SHARDING = "TP_SHARDING"
    DP_TP_SHARDING = "DP_TP_SHARDING"
    EP_SHARDING = "EP_SHARDING"
    EP_TP_SHARDING = "EP_TP_SHARDING"


_default_parameter_sharding_dimensions = {
    "word_embeddings.weight": 0,
    "lm_head.weight": 0,
    "query_key_value.weight": 0,
    "q_proj.weight": 0,
    "k_proj.weight": 0,
    "v_proj.weight": 0,
    "g_proj.weight": 0,
    "q_a_proj.weight": 0,
    "q_b_proj.weight": 0,
    "kv_a_proj_with_mqa.weight": 0,
    "kv_b_proj.weight": 0,
    "fused_qkv_a_proj_with_mqa.weight": 0,
    "dense.weight": 1,
    "o_proj.weight": 1,
    "g_norm.weight": 0,
    "query_key_value.bias": 0,
    "q_proj.bias": 0,
    "k_proj.bias": 0,
    "v_proj.bias": 0,
    "dense.bias": 0,
    "experts.w13_weight": 0,
    "experts.w2_weight": 1,
    "gate_up_proj.weight": 0,
    "gate_proj.weight": 0,
    "up_proj.weight": 0,
    "down_proj.weight": 1,
    "gate_up_proj.bias": 0,
    "gate_proj.bias": 0,
    "up_proj.bias": 0,
    "down_proj.bias": 0,
}


def get_default_sharding_dim(param_name):
    """Get sharding dimension using the last two parts of parameter name.
    Example usage:
    >>> get_default_sharding_dim("model.layers.0.attention.query_key_value.weight")
    0
    >>> get_default_sharding_dim("model.layers.0.attention.dense.weight")
    1
    >>> get_default_sharding_dim("model.layers.0.mlp.experts.w13_weight")
    1
    >>> get_default_sharding_dim("model.layers.0.mlp.experts.w2_weight")
    2
    >>> get_default_sharding_dim("model.layers.0.mlp.shared_experts.down_proj.weight")
    1
    >>> get_default_sharding_dim("model.layers.0.input_layernorm.weight")
    0
    >>> get_default_sharding_dim("model.layers.0.mlp.gate_up_proj.weight")
    0
    >>> get_default_sharding_dim("model.layers.0.mlp.gate_proj.weight")
    0
    >>> get_default_sharding_dim("model.layers.0.mlp.up_proj.weight")
    0
    >>> get_default_sharding_dim("model.layers.0.mlp.down_proj.weight")
    1
    """
    param_name = param_name.replace("_scale_inv", "")
    parts = param_name.split(".")
    if len(parts) >= 2:
        key = f"{parts[-2]}.{parts[-1]}"
        if key in _default_parameter_sharding_dimensions:
            return _default_parameter_sharding_dimensions[key]
    key = parts[-1]
    if key in _default_parameter_sharding_dimensions:
        return _default_parameter_sharding_dimensions[key]
    return 0


class ShardingStrategy:
    """
    Base class for defining sharding strategies for model parameters.
    Can be subclassed to customize behavior for special models.
    """

    def __init__(
        self,
        engine_name: str,
        enable_dp_attention,
        enable_dp_lm_head,
        moe_dense_tp_size,
        tp_size,
        ep_size,
        ep_tp_size,
        rank_info: RankInfo,
        device_backend: str | None = None,
        **kwargs,
    ) -> None:
        self.engine_name = engine_name
        self.enable_dp_attention = enable_dp_attention
        self.enable_dp_lm_head = enable_dp_lm_head
        self.moe_dense_tp_size = moe_dense_tp_size
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.ep_tp_size = ep_tp_size
        self.rank_info = rank_info

    def _maybe_adjust_sharding_dim(self, parameter_name: str, sharding_dim: int) -> int:
        return sharding_dim

    def get_attention_sharding_strategy(self, parameter_name, **kwargs):
        """
        Determine sharding strategy for attention parameters.
        Returns (ShardingType, num_shards).
        """
        sharding_dim = self._maybe_adjust_sharding_dim(
            parameter_name, get_default_sharding_dim(parameter_name)
        )
        if self.enable_dp_attention:
            attn_tp_size = self.rank_info.attn_tp_size
            if attn_tp_size > 1:
                return (ShardingType.DP_TP_SHARDING, sharding_dim, attn_tp_size)
            else:
                return (ShardingType.NO_SHARDING, sharding_dim, 1)
        else:
            tp_size = self.rank_info.tp_size
            if tp_size > 1:
                return (ShardingType.TP_SHARDING, sharding_dim, tp_size)
            else:
                return (ShardingType.NO_SHARDING, sharding_dim, 1)

    def get_embedding_sharding_strategy(self, *args, **kwargs):
        """
        By default, embedding sharding follows the attention sharding strategy.
        """
        return self.get_attention_sharding_strategy(*args, **kwargs)

    def get_mlp_sharding_strategy(self, parameter_name, **kwargs):
        """
        Determine sharding strategy for MLP parameters.
        Returns (ShardingType, num_shards).
        """
        sharding_dim = self._maybe_adjust_sharding_dim(
            parameter_name, get_default_sharding_dim(parameter_name)
        )
        tp_size = self.rank_info.tp_size
        if tp_size > 1:
            return (ShardingType.TP_SHARDING, sharding_dim, tp_size)
        return (ShardingType.NO_SHARDING, sharding_dim, 1)

    def get_shared_expert_sharding_strategy(self, parameter_name, **kwargs):
        """
        Determine sharding strategy for shared expert parameters.
        Returns (ShardingType, num_shards).
        """
        if self.engine_name == "mcore":
            sharding_dim = self._maybe_adjust_sharding_dim(
                parameter_name, get_default_sharding_dim(parameter_name)
            )
            if self.tp_size > 1:
                return (ShardingType.TP_SHARDING, sharding_dim, self.tp_size)
            else:
                return (ShardingType.NO_SHARDING, sharding_dim, 1)
        else:
            return self.get_expert_sharding_strategy(parameter_name, **kwargs)

    def get_expert_sharding_strategy(self, parameter_name, **kwargs):
        """
        Determine sharding strategy for expert model parameters.
        Returns (ShardingType, num_shards).
        """
        sharding_dim = self._maybe_adjust_sharding_dim(
            parameter_name, get_default_sharding_dim(parameter_name)
        )
        if self.ep_size > 1 and self.ep_tp_size > 1:
            return (ShardingType.EP_TP_SHARDING, sharding_dim, self.ep_tp_size)
        elif self.ep_size > 1:
            return (ShardingType.EP_SHARDING, sharding_dim, self.ep_size)
        elif self.tp_size > 1:
            return (ShardingType.TP_SHARDING, sharding_dim, self.tp_size)
        else:
            return (ShardingType.NO_SHARDING, sharding_dim, 1)

    def get_lm_head_sharding_strategy(self, parameter_name, **kwargs):
        """
        Determine sharding strategy for LM head parameters.
        Returns (ShardingType, num_shards).
        """
        sharding_dim = self._maybe_adjust_sharding_dim(
            parameter_name, get_default_sharding_dim(parameter_name)
        )
        if self.enable_dp_lm_head:
            attn_tp_size = self.rank_info.attn_tp_size
            if attn_tp_size > 1:
                return (ShardingType.DP_TP_SHARDING, sharding_dim, attn_tp_size)
            else:
                return (ShardingType.NO_SHARDING, sharding_dim, 1)
        else:
            tp_size = self.rank_info.tp_size
            if tp_size > 1:
                return (ShardingType.TP_SHARDING, sharding_dim, tp_size)
            else:
                return (ShardingType.NO_SHARDING, sharding_dim, 1)

    def get_sharding_strategy(self, parameter_name, **kwargs):
        """
        Determine the sharding strategy for a parameter based on its name and configuration.
        Returns (ShardingType, sharding_dim, num_shards).
        """
        tp_size = self.rank_info.tp_size
        if tp_size == 1:
            sharding_dim = self._maybe_adjust_sharding_dim(
                parameter_name, get_default_sharding_dim(parameter_name)
            )
            return (ShardingType.NO_SHARDING, sharding_dim, 1)
        if (
            "input_layernorm" in parameter_name
            or "post_attention_layernorm" in parameter_name
        ):
            return (ShardingType.NO_SHARDING, 0, 1)
        if "norm" in parameter_name:
            return (ShardingType.NO_SHARDING, 0, 1)
        if "embedding" in parameter_name or "embed_tokens" in parameter_name:
            return self.get_embedding_sharding_strategy(parameter_name, **kwargs)
        if "lm_head" in parameter_name:
            return self.get_lm_head_sharding_strategy(parameter_name, **kwargs)
        if "expert_bias" in parameter_name:
            return (ShardingType.NO_SHARDING, 0, 1)
        if "shared_experts" in parameter_name:
            return self.get_shared_expert_sharding_strategy(parameter_name, **kwargs)
        if "expert" in parameter_name:
            return self.get_expert_sharding_strategy(parameter_name, **kwargs)
        if "gate.weight" in parameter_name or "router.weight" in parameter_name:
            return (ShardingType.NO_SHARDING, 0, 1)
        if "mlp" in parameter_name:
            return self.get_mlp_sharding_strategy(parameter_name, **kwargs)
        if "attention" in parameter_name:
            return self.get_attention_sharding_strategy(parameter_name, **kwargs)
        sharding_dim = self._maybe_adjust_sharding_dim(
            parameter_name, get_default_sharding_dim(parameter_name)
        )
        return (ShardingType.TP_SHARDING, sharding_dim, tp_size)
