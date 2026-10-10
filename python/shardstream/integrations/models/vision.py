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

"""Qwen3-VL dense and MoE weight conversion support."""

from shardstream.metadata.sharding import (
    ShardingStrategy,
    ShardingType,
    get_default_sharding_dim,
)


class Qwen3VLShardingStrategy(ShardingStrategy):
    """Describe the vLLM/Megatron TP layout of Qwen3-VL vision weights."""

    def _vision_tp_strategy(self, sharding_dim: int):
        if self.enable_dp_attention:
            tp_size = self.rank_info.attn_tp_size
            sharding_type = ShardingType.DP_TP_SHARDING
        else:
            tp_size = self.rank_info.tp_size
            sharding_type = ShardingType.TP_SHARDING
        if tp_size > 1:
            return sharding_type, sharding_dim, tp_size
        return ShardingType.NO_SHARDING, sharding_dim, 1

    def get_sharding_strategy(self, parameter_name: str, **kwargs):
        if not parameter_name.startswith("model.visual."):
            return super().get_sharding_strategy(parameter_name, **kwargs)

        sharding_dim = get_default_sharding_dim(parameter_name)
        replicated_suffixes = (
            ".bias",
            ".norm.weight",
            ".norm.bias",
            ".norm1.weight",
            ".norm1.bias",
            ".norm2.weight",
            ".norm2.bias",
        )
        if parameter_name.startswith("model.visual.patch_embed.") or (
            parameter_name.endswith(replicated_suffixes)
            and not parameter_name.endswith(("qkv.bias", "linear_fc1.bias"))
        ):
            return ShardingType.NO_SHARDING, sharding_dim, 1

        if parameter_name.endswith("pos_embed.weight"):
            # Both Megatron and vLLM keep the learned position table replicated.
            return ShardingType.NO_SHARDING, 0, 1

        if parameter_name.endswith(
            (
                "attn.qkv.weight",
                "attn.qkv.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias",
                "merger.linear_fc1.weight",
                "merger.linear_fc1.bias",
            )
        ) or (
            ".deepstack_merger_list." in parameter_name
            and parameter_name.endswith(("linear_fc1.weight", "linear_fc1.bias"))
        ):
            return self._vision_tp_strategy(0)

        if parameter_name.endswith(
            (
                "attn.proj.weight",
                "mlp.linear_fc2.weight",
                "merger.linear_fc2.weight",
            )
        ) or (
            ".deepstack_merger_list." in parameter_name
            and parameter_name.endswith("linear_fc2.weight")
        ):
            return self._vision_tp_strategy(1)

        return ShardingType.NO_SHARDING, sharding_dim, 1
