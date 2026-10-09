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


from typing import Dict, List, Optional, Tuple

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import distributed as dist
from transformers import PretrainedConfig

from shardstream import logging
from shardstream._utils.common import divide
from shardstream.metadata.rank import RankInfo

logger = logging.getLogger(__name__)


def _get_head_size(tf_config: TransformerConfig) -> int:
    kv_channels = getattr(tf_config, "kv_channels", None)
    if kv_channels is not None:
        return kv_channels
    return divide(tf_config.hidden_size, tf_config.num_attention_heads)


def _cfg_get(tf_config, key: str, default=None):
    if tf_config is None:
        return default
    if isinstance(tf_config, dict):
        return tf_config.get(key, default)
    return getattr(tf_config, key, default)


def _normalize_pp_stage_layer_id_map(
    raw_map: Optional[Dict],
) -> Dict[Tuple[int, int], Dict[int, int]]:
    if not raw_map:
        return {}
    normalized: Dict[Tuple[int, int], Dict[int, int]] = {}
    for raw_key, raw_layer_map in raw_map.items():
        if isinstance(raw_key, tuple) and len(raw_key) == 2:
            pp_rank = int(raw_key[0])
            vp_stage = int(raw_key[1])
        elif isinstance(raw_key, str):
            if ":" not in raw_key:
                raise ValueError(
                    f"Invalid pp stage map key, expected '<pp_rank>:<vp_stage>': {raw_key}"
                )
            (pp_raw, vp_raw) = raw_key.split(":", 1)
            pp_rank = int(pp_raw)
            vp_stage = int(vp_raw)
        else:
            raise ValueError(f"Unsupported pp stage map key type: {type(raw_key)}")
        if not isinstance(raw_layer_map, dict):
            raise ValueError(
                f"Invalid pp stage map value, expected dict: key={raw_key}, value_type={type(raw_layer_map)}"
            )
        stage_map: Dict[int, int] = {}
        for local_layer_id, global_layer_id in raw_layer_map.items():
            stage_map[int(local_layer_id)] = int(global_layer_id)
        normalized[pp_rank, vp_stage] = stage_map
    return normalized


def _is_decoder_layer(layer_type) -> bool:
    if hasattr(layer_type, "name"):
        return str(layer_type.name).lower() == "decoder"
    normalized = str(layer_type).strip().lower()
    return (
        normalized == "decoder" or normalized == "t" or normalized.endswith(".decoder")
    )


def _normalize_pipeline_layout(
    pipeline_layout, pp_size: int
) -> Tuple[List[List[List[object]]], int]:
    layout = getattr(pipeline_layout, "layout", pipeline_layout)
    if not isinstance(layout, list) or not layout:
        raise ValueError("pipeline_model_parallel_layout must be a non-empty list")
    if isinstance(layout[0], list) and layout[0] and isinstance(layout[0][0], list):
        layout_2d = layout
    elif isinstance(layout[0], list):
        if len(layout) % pp_size != 0:
            raise ValueError(
                f"pipeline_model_parallel_layout stage count must be divisible by pp_size: stages={len(layout)}, pp_size={pp_size}"
            )
        vp_size = len(layout) // pp_size
        layout_2d = [
            [layout[vp * pp_size + pp_rank] for vp in range(vp_size)]
            for pp_rank in range(pp_size)
        ]
    else:
        raise ValueError(
            "pipeline_model_parallel_layout must contain stage layer lists"
        )
    if len(layout_2d) != pp_size:
        raise ValueError(
            f"pipeline_model_parallel_layout pp dimension mismatch: layout_pp={len(layout_2d)}, pp_size={pp_size}"
        )
    vp_size = len(layout_2d[0])
    if vp_size <= 0:
        raise ValueError("pipeline_model_parallel_layout has empty vp dimension")
    if any((len(pp_row) != vp_size for pp_row in layout_2d)):
        raise ValueError(
            "pipeline_model_parallel_layout has inconsistent vp dimension across pp ranks"
        )
    return (layout_2d, vp_size)


def _build_stage_layer_counts_from_layout(
    pipeline_layout, pp_size: int
) -> List[List[int]]:
    (layout_2d, vp_size) = _normalize_pipeline_layout(pipeline_layout, pp_size)
    stage_counts = [[0 for _ in range(vp_size)] for _ in range(pp_size)]
    for pp_rank in range(pp_size):
        for vp_stage in range(vp_size):
            stage_counts[pp_rank][vp_stage] = sum(
                (
                    1
                    for layer_type in layout_2d[pp_rank][vp_stage]
                    if _is_decoder_layer(layer_type)
                )
            )
    return stage_counts


def _build_stage_layer_counts_from_split_config(
    num_hidden_layers: int, rank_info: RankInfo, tf_config
) -> List[List[int]]:
    pp_size = max(int(rank_info.pp_size), 1)
    pp_ranks = list(range(pp_size))
    pp_rank_last = pp_size - 1
    first_layers = _cfg_get(tf_config, "num_layers_in_first_pipeline_stage", None)
    last_layers = _cfg_get(tf_config, "num_layers_in_last_pipeline_stage", None)
    account_embed = bool(
        _cfg_get(tf_config, "account_for_embedding_in_pipeline_split", False)
    )
    account_loss = bool(
        _cfg_get(tf_config, "account_for_loss_in_pipeline_split", False)
    )
    vp_size_cfg = _cfg_get(tf_config, "virtual_pipeline_model_parallel_size", None)
    vp_size = int(vp_size_cfg) if vp_size_cfg is not None and pp_size > 1 else 1
    if vp_size <= 0:
        raise ValueError(
            f"virtual_pipeline_model_parallel_size must be positive, got {vp_size}"
        )
    if (first_layers is not None or last_layers is not None) and (
        account_embed or account_loss
    ):
        raise ValueError(
            "num_layers_in_first/last_pipeline_stage cannot be mixed with account_for_embedding/loss_in_pipeline_split"
        )
    layers_per_pp_rank = {}
    if first_layers is not None or last_layers is not None:
        if first_layers is not None and int(first_layers) <= 0:
            raise ValueError(
                f"num_layers_in_first_pipeline_stage must be positive, got {first_layers}"
            )
        if last_layers is not None and int(last_layers) <= 0:
            raise ValueError(
                f"num_layers_in_last_pipeline_stage must be positive, got {last_layers}"
            )
        first_layers = int(first_layers or 0)
        last_layers = int(last_layers or 0)
        layers_to_distribute = num_hidden_layers - first_layers - last_layers
        middle_pp_size = pp_size
        if _cfg_get(tf_config, "num_layers_in_first_pipeline_stage", None) is not None:
            middle_pp_size -= 1
        if _cfg_get(tf_config, "num_layers_in_last_pipeline_stage", None) is not None:
            middle_pp_size -= 1
        if layers_to_distribute < 0:
            raise ValueError(
                f"Invalid uneven pipeline setup: specified first/last layers exceed total layers (total={num_hidden_layers}, first={first_layers}, last={last_layers})"
            )
        if middle_pp_size > 0 and layers_to_distribute % middle_pp_size != 0:
            raise ValueError(
                f"Uneven pipeline middle layers must be divisible by middle pp size: middle_layers={layers_to_distribute}, middle_pp_size={middle_pp_size}"
            )
        middle_layers = (
            layers_to_distribute // middle_pp_size if middle_pp_size > 0 else 0
        )
        for pp_rank in pp_ranks:
            value = middle_layers
            if (
                pp_rank == 0
                and _cfg_get(tf_config, "num_layers_in_first_pipeline_stage", None)
                is not None
            ):
                value = first_layers
            if (
                pp_rank == pp_rank_last
                and _cfg_get(tf_config, "num_layers_in_last_pipeline_stage", None)
                is not None
            ):
                value = last_layers
            layers_per_pp_rank[pp_rank] = value
    else:
        effective_layers = num_hidden_layers
        if account_embed:
            effective_layers += 1
        if account_loss:
            effective_layers += 1
        if effective_layers % pp_size != 0:
            raise ValueError(
                f"PP split is not divisible under account_for_embedding/loss settings: effective_layers={effective_layers}, pp_size={pp_size}, num_hidden_layers={num_hidden_layers}, account_for_embedding={account_embed}, account_for_loss={account_loss}"
            )
        layers_per_pp = effective_layers // pp_size
        for pp_rank in pp_ranks:
            layers_per_pp_rank[pp_rank] = layers_per_pp
    stage_counts = [[0 for _ in range(vp_size)] for _ in range(pp_size)]
    for pp_rank in pp_ranks:
        layers_this_pp = int(layers_per_pp_rank[pp_rank])
        if layers_this_pp < 0:
            raise ValueError(
                f"Negative layer count on pp_rank={pp_rank}: {layers_this_pp}"
            )
        if vp_size > 1 and layers_this_pp % vp_size != 0:
            raise ValueError(
                f"Layers per pp rank must be divisible by virtual pipeline size: pp_rank={pp_rank}, layers={layers_this_pp}, vp_size={vp_size}"
            )
        per_vp = layers_this_pp // vp_size if vp_size > 1 else layers_this_pp
        for vp_stage in range(vp_size):
            stage_counts[pp_rank][vp_stage] = per_vp
    if account_embed:
        stage_counts[0][0] -= 1
    if account_loss:
        stage_counts[pp_rank_last][vp_size - 1] -= 1
    for pp_rank in pp_ranks:
        for vp_stage in range(vp_size):
            if stage_counts[pp_rank][vp_stage] < 0:
                raise ValueError(
                    f"Negative decoder layer count after account_for_* adjustment: pp_rank={pp_rank}, vp_stage={vp_stage}, count={stage_counts[pp_rank][vp_stage]}"
                )
    return stage_counts


def _resolve_pp_stage_global_layer_ids(
    rank_info: RankInfo,
    hf_config: PretrainedConfig,
    tf_config: TransformerConfig,
    vp_stage: int = None,
) -> List[int]:
    num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        num_hidden_layers = _cfg_get(tf_config, "num_layers", None)
    if num_hidden_layers is None:
        raise ValueError(
            "num_hidden_layers is required for PP layer mapping (missing in hf_config and tf_config)"
        )
    num_hidden_layers = int(num_hidden_layers)
    pp_size = max(int(rank_info.pp_size), 1)
    pp_rank = int(rank_info.pp_rank)
    if pp_rank < 0 or pp_rank >= pp_size:
        raise ValueError(f"Invalid pp_rank={pp_rank} for pp_size={pp_size}")
    pipeline_layout = _cfg_get(tf_config, "pipeline_model_parallel_layout", None)
    if pipeline_layout is not None:
        stage_counts = _build_stage_layer_counts_from_layout(pipeline_layout, pp_size)
    else:
        stage_counts = _build_stage_layer_counts_from_split_config(
            num_hidden_layers, rank_info, tf_config
        )
    vp_size = len(stage_counts[0]) if stage_counts else 1
    stage_vp = 0 if vp_stage is None else int(vp_stage)
    if stage_vp < 0 or stage_vp >= vp_size:
        raise ValueError(
            f"Invalid vp_stage={stage_vp} for vp_size={vp_size}, pp_rank={pp_rank}"
        )
    start_offset = 0
    stage_layer_count = stage_counts[pp_rank][stage_vp]
    for curr_vp in range(vp_size):
        for curr_pp in range(pp_size):
            if curr_pp == pp_rank and curr_vp == stage_vp:
                break
            start_offset += stage_counts[curr_pp][curr_vp]
        else:
            continue
        break
    total_layers = sum((sum(row) for row in stage_counts))
    if total_layers != num_hidden_layers:
        raise ValueError(
            f"Resolved PP stage layer counts do not match num_hidden_layers: resolved_total={total_layers}, num_hidden_layers={num_hidden_layers}, pp_size={pp_size}, stage_counts={stage_counts}"
        )
    return list(range(start_offset, start_offset + stage_layer_count))


def _process_mcore_pp_name(
    name: str,
    rank_info: RankInfo,
    hf_config: PretrainedConfig,
    tf_config: TransformerConfig,
    vp_stage: int = None,
    pp_stage_layer_id_map: Optional[Dict[Tuple[int, int], Dict[int, int]]] = None,
) -> str:
    """
    Process the name of a parameter to remove the pipeline parallel rank.
    """
    if "layers." in name:
        (left, remains) = name.rsplit(".layers.", 1)
        splits = remains.split(".")
        local_layer_id = int(splits[0])
        if not pp_stage_layer_id_map:
            return name
        stage_vp = 0 if vp_stage is None else int(vp_stage)
        stage_key = (int(rank_info.pp_rank), stage_vp)
        layer_map = pp_stage_layer_id_map.get(stage_key)
        if layer_map is None:
            raise ValueError(
                f"Missing pp stage layer map for current rank/stage: pp_rank={rank_info.pp_rank}, vp_stage={stage_vp}, name={name}"
            )
        if local_layer_id not in layer_map:
            stage_layer_ids = sorted(layer_map.keys())
            raise ValueError(
                f"Local layer id missing in pp stage layer map: local_layer_id={local_layer_id}, pp_rank={rank_info.pp_rank}, pp_size={rank_info.pp_size}, vp_stage={stage_vp}, known_local_ids={stage_layer_ids}, name={name}"
            )
        global_layer_id = layer_map[local_layer_id]
        remaining_parts = ".".join(splits[1:])
        return f"{left}.layers.{global_layer_id}.{remaining_parts}"
    else:
        return name


class McoreToHFWeightConverter:
    def __init__(
        self,
        hf_config: PretrainedConfig,
        rank_info: RankInfo,
        infer_conf: Dict,
        tf_config: TransformerConfig,
    ):
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.rank_info = rank_info
        self.infer_conf = infer_conf
        self._pp_stage_layer_id_map = _normalize_pp_stage_layer_id_map(
            infer_conf.get("train_pp_stage_layer_id_map")
        )
        self.router_dtype = infer_conf.get("router_dtype", "bf16")
        if self.router_dtype == "bf16":
            self.router_dtype = torch.bfloat16
        elif self.router_dtype == "fp16":
            self.router_dtype = torch.float16
        elif self.router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported router dtype: {self.router_dtype}")
        if "infer_atten_tp_size" not in infer_conf:
            raise ValueError("infer_atten_tp_size must be specified")
        self.infer_atten_tp_size = infer_conf["infer_atten_tp_size"]

    def _convert_attention_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        if "self_attention.linear_qkv.weight" in name:
            if self._fuse_qkv(name):
                parameter = convert_qkv_weight_along_tp_attention(
                    parameter,
                    self.infer_atten_tp_size,
                    self.tf_config,
                    train_tp_rank=self.rank_info.attn_tp_rank,
                    train_tp_size=self.rank_info.attn_tp_size,
                )
                return [("self_attn.qkv_proj.weight", parameter)]
            else:
                shape0 = parameter.shape[0]
                stride = shape0 // 3
                return [
                    ("self_attn.q_proj.weight", parameter.narrow(0, 0, stride)),
                    ("self_attn.k_proj.weight", parameter.narrow(0, stride, stride)),
                    (
                        "self_attn.v_proj.weight",
                        parameter.narrow(0, 2 * stride, stride),
                    ),
                ]
        elif "self_attention.linear_qkv.bias" in name:
            if self._fuse_qkv(name):
                parameter = convert_qkv_bias_along_tp_attention(
                    parameter,
                    self.infer_atten_tp_size,
                    self.tf_config,
                    train_tp_rank=self.rank_info.attn_tp_rank,
                    train_tp_size=self.rank_info.attn_tp_size,
                )
                return [("self_attn.qkv_proj.bias", parameter)]
            else:
                (query, key, value) = transform_mcore_qkv_bias(
                    parameter, self.tf_config
                )
                return [
                    ("self_attn.q_proj.bias", query),
                    ("self_attn.k_proj.bias", key),
                    ("self_attn.v_proj.bias", value),
                ]
        elif "self_attention.linear_qkv.layer_norm_weight" in name:
            return [("input_layernorm.weight", parameter)]
        elif "self_attention.linear_proj.weight" in name:
            return [("self_attn.o_proj.weight", parameter)]
        elif "self_attention.linear_proj.bias" in name:
            return [("self_attn.o_proj.bias", parameter)]
        elif "self_attention.q_layernorm.weight" in name:
            return [("self_attn.q_norm.weight", parameter)]
        elif "self_attention.k_layernorm.weight" in name:
            return [("self_attn.k_norm.weight", parameter)]
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _convert_linear(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        converted: List[Tuple[str, torch.Tensor]]
        if "linear_fc1.weight" in name:
            if self._fuse_gate_up_proj(name):
                converted = [("gate_up_proj.weight", parameter)]
            else:
                converted = [
                    (
                        "gate_proj.weight",
                        parameter.narrow(0, 0, parameter.shape[0] // 2),
                    ),
                    (
                        "up_proj.weight",
                        parameter.narrow(
                            0, parameter.shape[0] // 2, parameter.shape[0] // 2
                        ),
                    ),
                ]
        elif "linear_fc1.bias" in name:
            if self._fuse_gate_up_proj(name):
                converted = [("gate_up_proj.bias", parameter)]
            else:
                converted = [
                    ("gate_proj.bias", parameter.narrow(0, 0, parameter.shape[0] // 2)),
                    (
                        "up_proj.bias",
                        parameter.narrow(
                            0, parameter.shape[0] // 2, parameter.shape[0] // 2
                        ),
                    ),
                ]
        elif "linear_fc2.weight" in name:
            converted = [("down_proj.weight", parameter)]
        elif "linear_fc2.bias" in name:
            converted = [("down_proj.bias", parameter)]
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return converted

    def _convert_gate(
        self, name: str, parameter: torch.Tensor
    ) -> Tuple[str, torch.Tensor]:
        assert "router" in name or "gate." in name, (
            f"Unsupported parameter name: {name}"
        )
        return ("mlp.gate.weight", parameter.to(self.router_dtype))

    def _convert_mlp_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> List[Tuple[str, torch.Tensor]]:
        assert "attention" not in name, (
            f"{name} shouble be hanled in _convert_attention_param"
        )
        if "pre_mlp_layernorm" in name or "linear_fc1.layer_norm_weight" in name:
            return [("post_attention_layernorm.weight", parameter)]
        if "input_layernorm.weight" in name:
            return [("input_layernorm.weight", parameter)]
        elif "shared_experts.gate_weight" in name:
            return [("mlp.shared_expert_gate.weight", parameter)]
        elif (
            "shared_experts.linear_fc1.weight" in name
            or "shared_experts.linear_fc1.bias" in name
        ):
            return [
                (f"mlp.shared_experts.{name}", param)
                for (name, param) in self._convert_linear(name, parameter)
            ]
        elif (
            "shared_experts.linear_fc2.weight" in name
            or "shared_experts.linear_fc2.bias" in name
        ):
            return [
                (f"mlp.shared_experts.{name}", param)
                for (name, param) in self._convert_linear(name, parameter)
            ]
        elif "mlp.experts." in name:
            if "local_experts" in name:
                local_expert_id = int(name.rsplit(".", 3)[-3])
            else:
                local_expert_id = int(name.rsplit("weight", 1)[-1])
            num_experts = self.hf_config.num_experts
            num_experts_per_partition = num_experts // self.rank_info.ep_size
            expert_id = (
                local_expert_id + self.rank_info.ep_rank * num_experts_per_partition
            )
            return [
                (f"mlp.experts.{expert_id}.{name}", param)
                for (name, param) in self._convert_linear(name, parameter)
            ]
        elif (
            "linear_fc1.weight" in name
            or "linear_fc2.weight" in name
            or "linear_fc1.bias" in name
            or ("linear_fc2.bias" in name)
        ):
            return [
                (f"mlp.{name}", param)
                for (name, param) in self._convert_linear(name, parameter)
            ]
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _convert_expert_bias_param(
        self, name: str, parameter: torch.Tensor, layer_number: str
    ) -> Tuple[str, torch.Tensor]:
        """Convert bias parameters"""
        if "expert_bias" in name:
            return ("mlp.gate.expert_bias", parameter.to(torch.float32))
        else:
            raise NotImplementedError(f"Unsupported bias parameter name: {name}")

    def _fuse_qkv(self, name: str) -> bool:
        """Override this method to control QKV fusion behavior"""
        return True

    def _fuse_gate_up_proj(self, name: str) -> bool:
        return False

    def _convert_lm_head_param(
        self, name: str, parameter: torch.Tensor
    ) -> List[Tuple[str, torch.Tensor]]:
        if getattr(self.hf_config, "norm_head", False):
            import torch.nn.functional as F

            parameter = F.normalize(parameter, dim=0, p=2, eps=1e-07)
        return [("lm_head.weight", parameter)]

    @staticmethod
    def _normalize_attn_name(name: str) -> str:
        replacements = [
            ("self_attn.qkv_proj", "attention.query_key_value_proj"),
            ("self_attn.o_proj", "attention.dense"),
            ("self_attn.q_norm", "attention.query_layernorm"),
            ("self_attn.k_norm", "attention.key_layernorm"),
        ]
        for old, new in replacements:
            if old in name:
                name = name.replace(old, new)
        name = name.replace("query_key_value_proj_proj", "query_key_value_proj")
        return name

    def _canonicalize_source_name(self, name: str, vp_stage: int = None) -> str:
        name = name.replace("module.", "")
        return _process_mcore_pp_name(
            name,
            self.rank_info,
            self.hf_config,
            self.tf_config,
            vp_stage=vp_stage,
            pp_stage_layer_id_map=self._pp_stage_layer_id_map,
        )

    @torch.no_grad()
    def convert_param(
        self, name: str, parameter: torch.Tensor, vp_stage: int = None
    ) -> List[Tuple[str, torch.Tensor]]:
        name = self._canonicalize_source_name(name, vp_stage)
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        if name in direct_name_mapping:
            return [(direct_name_mapping[name], parameter)]
        if "output_layer.weight" in name:
            return self._convert_lm_head_param(name, parameter)
        name = name.replace("decoder.layers.", "")
        (layer_number, remaining_name) = name.split(".", 1)
        if "self_attention" in remaining_name:
            converted = []
            for attn_name, param in self._convert_attention_param(
                remaining_name, parameter, layer_number
            ):
                attn_name = self._normalize_attn_name(attn_name)
                converted.append((f"model.layers.{layer_number}.{attn_name}", param))
            return converted
        elif "mlp" in remaining_name:
            if "mlp.gate.weight" in name or "mlp.router.weight" in name:
                (name, param) = self._convert_gate(name, parameter)
                return [(f"model.layers.{layer_number}.{name}", param)]
            elif "expert_bias" in name:
                (name, param) = self._convert_expert_bias_param(
                    name, parameter, layer_number
                )
                return [(f"model.layers.{layer_number}.{name}", param)]
            return [
                (f"model.layers.{layer_number}.{name}", param)
                for (name, param) in self._convert_mlp_param(
                    remaining_name, parameter, layer_number
                )
            ]
        else:
            raise NotImplementedError(f"Unsupported parameter name: {name}")


def get_full_tensor(weight: torch.Tensor, dim: int = 0):
    from megatron.core import parallel_state as mpu

    train_tp_size = mpu.get_tensor_model_parallel_world_size()
    if train_tp_size != 1:
        tp_group = mpu.get_tensor_model_parallel_group()
        new_v = [torch.zeros_like(weight) for i in range(train_tp_size)]
        dist.all_gather(new_v, weight, group=tp_group, async_op=False)
        weight = torch.cat(new_v, dim=dim)
    return weight


def transform_mcore_qkv_weight(weight: torch.Tensor, tf_config: TransformerConfig):
    """
    Megatron QKV is alternately packed, fused QKV is packed consecutively.
                    tp0                tp1
                               │
               ┌───────┬───┬───│───────┬───┬───┐
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
    mcore      │   Q   │ K │ V │   Q   │ K │ V │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               │       │   │   │       │   │   │
               └───┬───┴─┬─┴─┬─│───┬───┴─┬─┴──┬┘
                   │     │   │ │   │     │    │
                   │     │   └─┴───┼─────┼┐   │
                   │     │┌────────┘     ││   │
                   │     ││              ││   │
                   │     └┼─────┐   ┌────┘│   │
                   │      │     │   │     │   │
               ┌───▼──────▼────┬▼───▼──┬──▼───▼┐
               │  Q0      Q1   │K0  K1 │  V0 V1│
               │               │       │       │
               │               │       │       │
    bailing    │       Q       │   K   │   V   │
               │               │       │       │
               │               │       │       │
               │               │       │       │
               └───────────────┴───────┴───────┘
    """
    weight = get_full_tensor(weight, dim=0)
    hidden_size = tf_config.hidden_size
    total_num_heads = tf_config.num_attention_heads
    total_num_kv_heads = tf_config.num_query_groups
    head_size = _get_head_size(tf_config)
    each_kv_size = head_size
    each_query_size = head_size * divide(total_num_heads, total_num_kv_heads)
    actual_size = weight.shape[0]
    expected_compact_size = (each_query_size + 2 * each_kv_size) * total_num_kv_heads
    expected_replicated_size = 3 * hidden_size
    expected_qwen3_gqa_size = hidden_size * 2 + 2 * hidden_size
    if actual_size == expected_compact_size:
        query_list = []
        key_list = []
        value_list = []
        for qkv in torch.chunk(weight, total_num_kv_heads, dim=0):
            (q, k, v) = qkv.split([each_query_size, each_kv_size, each_kv_size], dim=0)
            query_list.append(q)
            key_list.append(k)
            value_list.append(v)
        all_query = torch.cat(query_list, dim=0)
        all_key = torch.cat(key_list, dim=0)
        all_value = torch.cat(value_list, dim=0)
    elif actual_size == expected_replicated_size:
        (q, k, v) = weight.split([hidden_size, hidden_size, hidden_size], dim=0)
        heads_per_kv_group = divide(total_num_heads, total_num_kv_heads)
        k_heads = k.reshape(total_num_heads, head_size, -1)
        v_heads = v.reshape(total_num_heads, head_size, -1)
        k_unique = []
        v_unique = []
        for i in range(total_num_kv_heads):
            k_unique.append(k_heads[i * heads_per_kv_group])
            v_unique.append(v_heads[i * heads_per_kv_group])
        all_key = torch.cat(k_unique, dim=0).reshape(-1, k.shape[-1])
        all_value = torch.cat(v_unique, dim=0).reshape(-1, v.shape[-1])
        all_query = q
    elif actual_size == expected_qwen3_gqa_size:
        query_list = []
        key_list = []
        value_list = []
        head_size = divide(hidden_size, total_num_kv_heads)
        each_kv_size = head_size
        each_query_size = head_size * divide(total_num_heads, total_num_kv_heads)
        for qkv in torch.chunk(weight, total_num_kv_heads, dim=0):
            (q, k, v) = qkv.split([each_query_size, each_kv_size, each_kv_size], dim=0)
            query_list.append(q)
            key_list.append(k)
            value_list.append(v)
        all_query = torch.cat(query_list, dim=0)
        all_key = torch.cat(key_list, dim=0)
        all_value = torch.cat(value_list, dim=0)
    else:
        raise ValueError(
            f"QKV weight size mismatch - unsupported format:\n  Actual weight shape[0]: {actual_size}\n  Expected compact GQA size: {expected_compact_size}\n  Expected replicated size: {expected_replicated_size}\n  Config: hidden_size={hidden_size}, num_heads={total_num_heads}, num_kv_heads={total_num_kv_heads}, head_size={head_size}\n  Per-group sizes: query={each_query_size}, kv={each_kv_size}"
        )
    return (all_query, all_key, all_value)


def convert_qkv_weight_along_tp_attention(
    weight: torch.Tensor,
    infer_atten_tp_size: int,
    tf_config: TransformerConfig,
    train_tp_rank: int | None = None,
    train_tp_size: int | None = None,
):
    """
    fused QKV: The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.
    """
    total_num_kv_heads = tf_config.num_query_groups
    if infer_atten_tp_size >= total_num_kv_heads:
        num_kv_head_replicas = divide(infer_atten_tp_size, total_num_kv_heads)
    else:
        num_kv_head_replicas = 1
    (query, key, value) = transform_mcore_qkv_weight(weight, tf_config)
    query_shards = query.chunk(infer_atten_tp_size, dim=0)
    if infer_atten_tp_size >= total_num_kv_heads:
        key_chunks = key.chunk(total_num_kv_heads, dim=0)
        key_shards = [k for k in key_chunks for _ in range(num_kv_head_replicas)]
        value_chunks = value.chunk(total_num_kv_heads, dim=0)
        value_shards = [v for v in value_chunks for _ in range(num_kv_head_replicas)]
    else:
        key_shards = key.chunk(infer_atten_tp_size, dim=0)
        value_shards = value.chunk(infer_atten_tp_size, dim=0)
    qkv_tp_groups = []
    for query_shard, key_shard, value_shard in zip(
        query_shards, key_shards, value_shards
    ):
        qkv_tp_groups.append(query_shard)
        qkv_tp_groups.append(key_shard)
        qkv_tp_groups.append(value_shard)
    merged = torch.cat(qkv_tp_groups, dim=0)
    if train_tp_size and train_tp_size > 1:
        if train_tp_rank is None:
            raise ValueError("train_tp_rank is required when train_tp_size > 1")
        shards = torch.chunk(merged, train_tp_size, dim=0)
        if train_tp_rank >= len(shards):
            raise ValueError(
                f"train_tp_rank {train_tp_rank} out of range for tp_size {train_tp_size}"
            )
        return shards[train_tp_rank]
    return merged


def transform_mcore_qkv_bias(bias: torch.Tensor, tf_config: TransformerConfig):
    """
    Transform Megatron QKV bias for grouped-query attention.
    Similar to transform_mcore_qkv_weight but for bias parameters.
    """
    bias = get_full_tensor(bias, dim=0)
    hidden_size = tf_config.hidden_size
    total_num_heads = tf_config.num_attention_heads
    total_num_kv_heads = tf_config.num_query_groups
    head_size = _get_head_size(tf_config)
    each_kv_size = head_size
    each_query_size = head_size * divide(total_num_heads, total_num_kv_heads)
    actual_size = bias.shape[0]
    expected_compact_size = (each_query_size + 2 * each_kv_size) * total_num_kv_heads
    expected_replicated_size = 3 * hidden_size
    if actual_size == expected_compact_size:
        query_list = []
        key_list = []
        value_list = []
        for qkv in torch.chunk(bias, total_num_kv_heads, dim=0):
            (q, k, v) = qkv.split([each_query_size, each_kv_size, each_kv_size], dim=0)
            query_list.append(q)
            key_list.append(k)
            value_list.append(v)
        all_query = torch.cat(query_list, dim=0)
        all_key = torch.cat(key_list, dim=0)
        all_value = torch.cat(value_list, dim=0)
    elif actual_size == expected_replicated_size:
        (q, k, v) = bias.split([hidden_size, hidden_size, hidden_size], dim=0)
        heads_per_kv_group = divide(total_num_heads, total_num_kv_heads)
        k_heads = k.reshape(total_num_heads, head_size)
        v_heads = v.reshape(total_num_heads, head_size)
        k_unique = []
        v_unique = []
        for i in range(total_num_kv_heads):
            k_unique.append(k_heads[i * heads_per_kv_group])
            v_unique.append(v_heads[i * heads_per_kv_group])
        all_key = torch.cat(k_unique, dim=0).reshape(-1)
        all_value = torch.cat(v_unique, dim=0).reshape(-1)
        all_query = q
    else:
        raise ValueError(
            f"QKV bias size mismatch - unsupported format:\n  Actual bias shape[0]: {actual_size}\n  Expected compact GQA size: {expected_compact_size}\n  Expected replicated size: {expected_replicated_size}\n  Config: hidden_size={hidden_size}, num_heads={total_num_heads}, num_kv_heads={total_num_kv_heads}, head_size={head_size}\n  Per-group sizes: query={each_query_size}, kv={each_kv_size}"
        )
    return (all_query, all_key, all_value)


def convert_qkv_bias_along_tp_attention(
    bias: torch.Tensor,
    infer_atten_tp_size: int,
    tf_config: TransformerConfig,
    train_tp_rank: int | None = None,
    train_tp_size: int | None = None,
):
    """
    Convert QKV bias for fused format with TP attention.
    Similar to convert_qkv_weight_along_tp_attention but for bias parameters.
    """
    total_num_kv_heads = tf_config.num_query_groups
    if infer_atten_tp_size >= total_num_kv_heads:
        num_kv_head_replicas = divide(infer_atten_tp_size, total_num_kv_heads)
    else:
        num_kv_head_replicas = 1
    (query, key, value) = transform_mcore_qkv_bias(bias, tf_config)
    query_shards = query.chunk(infer_atten_tp_size, dim=0)
    if infer_atten_tp_size >= total_num_kv_heads:
        key_chunks = key.chunk(total_num_kv_heads, dim=0)
        key_shards = [k for k in key_chunks for _ in range(num_kv_head_replicas)]
        value_chunks = value.chunk(total_num_kv_heads, dim=0)
        value_shards = [v for v in value_chunks for _ in range(num_kv_head_replicas)]
    else:
        key_shards = key.chunk(infer_atten_tp_size, dim=0)
        value_shards = value.chunk(infer_atten_tp_size, dim=0)
    qkv_tp_groups = []
    for query_shard, key_shard, value_shard in zip(
        query_shards, key_shards, value_shards
    ):
        qkv_tp_groups.append(query_shard)
        qkv_tp_groups.append(key_shard)
        qkv_tp_groups.append(value_shard)
    merged = torch.cat(qkv_tp_groups, dim=0)
    if train_tp_size and train_tp_size > 1:
        if train_tp_rank is None:
            raise ValueError("train_tp_rank is required when train_tp_size > 1")
        shards = torch.chunk(merged, train_tp_size, dim=0)
        if train_tp_rank >= len(shards):
            raise ValueError(
                f"train_tp_rank {train_tp_rank} out of range for tp_size {train_tp_size}"
            )
        return shards[train_tp_rank]
    return merged


def _to_local_mcore_tensor(parameter: torch.Tensor) -> torch.Tensor:
    """Return the rank-local payload for Megatron DTensor parameters.

    During optimizer phases Megatron-FSDP swaps model parameters for DP-sharded
    DTensors and keeps the forward-ready tensor-parallel weight in ``orig_param``.
    ShardStream must publish that TP-local weight: the DTensor's public shape is global
    while ``to_local()`` is only an uneven flat DP shard. Plain TP DTensors do
    not have ``orig_param`` and can be localized directly.
    """
    original = getattr(parameter, "orig_param", None)
    if isinstance(original, torch.Tensor):
        parameter = original
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return parameter
    if isinstance(parameter, DTensor):
        return parameter.to_local()
    return parameter


def get_mcore_model_parameters(model) -> Dict[str, torch.Tensor]:
    params_dict = {
        name: _to_local_mcore_tensor(param)
        for (name, param) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if "expert_bias" in name:
            params_dict[name] = _to_local_mcore_tensor(param)
    return params_dict
