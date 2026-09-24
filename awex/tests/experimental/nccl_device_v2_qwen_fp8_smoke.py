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

"""Validate 128x128 block-wise FP8 transfer with one mounted Qwen weight."""

from __future__ import annotations  # noqa: I001

import argparse
import json
import os
from pathlib import Path

from awex.converter.weights_converter import per_block_cast_to_fp8
from awex.transfer.nccl_device_v2 import _load_extension
from safetensors import safe_open
import torch
import torch.distributed as dist

_DEFAULT_PARAMETER = "model.layers.0.mlp.experts.0.gate_proj.weight"
_FP8_E4M3_CODE = 4
_FLOAT32_CODE = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--parameter", default=_DEFAULT_PARAMETER)
    return parser.parse_args()


def _load_weight(model_path: Path, parameter: str) -> torch.Tensor:
    with (model_path / "model.safetensors.index.json").open() as stream:
        weight_map = json.load(stream)["weight_map"]
    try:
        shard = weight_map[parameter]
    except KeyError as error:
        raise ValueError(
            f"Parameter is not present in the model: {parameter}"
        ) from error
    with safe_open(model_path / shard, framework="pt", device="cpu") as weights:
        tensor = weights.get_tensor(parameter)
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Expected a BF16 source tensor, got {tensor.dtype}")
    return tensor.contiguous()


def _broadcast_unique_id(extension: object, rank: int) -> bytes:
    values = [extension.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("Qwen FP8 smoke test requires exactly two ranks")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    source = _load_weight(Path(args.model_path), args.parameter)
    expected_weight, expected_scale = per_block_cast_to_fp8(source, False)
    expected_weight = expected_weight.cuda()
    expected_scale = expected_scale.cuda()
    if rank == 0:
        weight = source.cuda()
        scale = torch.empty_like(expected_scale)
    else:
        weight = torch.empty_like(expected_weight)
        scale = torch.empty_like(expected_scale)
    tensors = [weight, scale]
    extension = _load_extension()
    unique_id = _broadcast_unique_id(extension, rank)
    peer = 1 - rank
    rows, cols = source.shape
    weight_numel = source.numel()
    scale_numel = expected_scale.numel()
    handle = extension.create(
        unique_id,
        world_size,
        rank,
        local_rank,
        60_000,
        64,
        8,
        512 * 1024,
        4 * 1024 * 1024,
    )
    try:
        launches = []
        for sequence in (1, 2):
            if rank == 1:
                weight.zero_()
                scale.zero_()
            dist.barrier()
            launches.append(
                dict(
                    extension.launch(
                        handle,
                        tensors,
                        [weight_numel, scale_numel * 4],
                        [0, 0],
                        [cols * weight.element_size(), scale_numel * 4],
                        [cols * weight.element_size(), scale_numel * 4],
                        [_FP8_E4M3_CODE, _FLOAT32_CODE],
                        [1, 4],
                        [scale, scale],
                        [1, 0] if rank == 0 else [0, 0],
                        [rows, 0] if rank == 0 else [0, 0],
                        [cols, 0] if rank == 0 else [0, 0],
                        [0, 0],
                        [0, 0],
                        [scale.stride(0), 0] if rank == 0 else [0, 0],
                        [128, 0] if rank == 0 else [0, 0],
                        [128, 0] if rank == 0 else [0, 0],
                        [peer, peer],
                        [0, 1],
                        [2, 0] if rank == 1 else [0, 2],
                        rank == 0,
                        sequence,
                    )
                )
            )
            dist.barrier()
            if rank == 1 and not torch.equal(weight.float(), expected_weight.float()):
                difference = (
                    (weight.float() - expected_weight.float()).abs().max().item()
                )
                raise AssertionError(
                    f"Qwen block-wise FP8 weight mismatch: max_abs={difference}"
                )
            if rank == 1 and not torch.equal(scale, expected_scale):
                difference = (scale - expected_scale).abs().max().item()
                raise AssertionError(
                    f"Qwen block-wise FP8 scale mismatch: max_abs={difference}"
                )

        metrics = launches[-1]
        expected_wire_bytes = weight_numel + scale_numel * 4
        expected_tensor_bytes = weight_numel * weight.element_size() + scale_numel * 4
        if metrics["wire_bytes"] != expected_wire_bytes:
            raise AssertionError(f"Unexpected block-wise wire bytes: {metrics}")
        if metrics["tensor_bytes"] != expected_tensor_bytes:
            raise AssertionError(f"Unexpected block-wise tensor bytes: {metrics}")
        expected_quant_tasks = 1 if rank == 0 else 0
        if metrics["blockwise_fp8_tasks"] != expected_quant_tasks:
            raise AssertionError(f"Unexpected block-wise task count: {metrics}")
        if metrics["blockwise_fp8_matrix_count"] != expected_quant_tasks:
            raise AssertionError(f"Unexpected block-wise matrix count: {metrics}")
        if not metrics["plan_cache_hit"]:
            raise AssertionError(f"Second launch missed the plan cache: {metrics}")
        dist.barrier()
        print(
            f"rank={rank} parameter={args.parameter} shape={tuple(source.shape)} "
            f"qwen_block_fp8={metrics}",
            flush=True,
        )
    finally:
        extension.destroy(handle)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
