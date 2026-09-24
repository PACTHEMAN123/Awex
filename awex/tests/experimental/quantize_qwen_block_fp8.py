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

"""Create a vLLM-loadable Qwen 128x128 block-wise FP8 checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from awex.converter.weights_converter import per_block_cast_to_fp8

_QUANTIZED_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _should_quantize(name: str, tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and name.endswith(_QUANTIZED_SUFFIXES)


def _copy_auxiliary_files(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".safetensors" or path.name == "config.json":
            continue
        shutil.copyfile(path, destination / path.name)


def _write_config(source: Path, destination: Path) -> None:
    with (source / "config.json").open() as stream:
        config = json.load(stream)
    config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "ignored_layers": ["lm_head"],
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    with (destination / "config.json").open("w") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _quantize_shard(
    source_path: Path,
    destination_path: Path,
    device: torch.device,
) -> tuple[dict[str, str], int, int]:
    tensors: dict[str, torch.Tensor] = {}
    weight_map = {}
    quantized_count = 0
    total_size = 0
    with safe_open(source_path, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        for name in source.keys():
            tensor = source.get_tensor(name)
            if _should_quantize(name, tensor):
                quantized, scale = per_block_cast_to_fp8(
                    tensor.to(device=device, non_blocking=False), False
                )
                tensors[name] = quantized.cpu()
                scale_name = f"{name}_scale_inv"
                tensors[scale_name] = scale.cpu()
                weight_map[scale_name] = destination_path.name
                quantized_count += 1
            else:
                tensors[name] = tensor
            weight_map[name] = destination_path.name

    for tensor in tensors.values():
        total_size += int(tensor.numel()) * int(tensor.element_size())
    temporary_path = destination_path.with_suffix(".safetensors.incomplete")
    save_file(tensors, temporary_path, metadata=metadata)
    temporary_path.replace(destination_path)
    del tensors
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return weight_map, total_size, quantized_count


def main() -> None:
    args = _parse_args()
    source = args.input.resolve()
    destination = args.output.resolve()
    if source == destination:
        raise ValueError("Input and output checkpoint directories must differ")
    with (source / "model.safetensors.index.json").open() as stream:
        source_index = json.load(stream)
    shard_names = sorted(set(source_index["weight_map"].values()))

    destination.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(destination.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty (use --overwrite): {destination}"
        )
    output_index_path = destination / "model.safetensors.index.json"
    if output_index_path.exists():
        output_index_path.unlink()
    _copy_auxiliary_files(source, destination)
    _write_config(source, destination)

    device = torch.device(args.device)
    weight_map = {}
    total_size = 0
    total_quantized = 0
    for index, shard_name in enumerate(shard_names, start=1):
        print(f"[{index}/{len(shard_names)}] quantizing {shard_name}", flush=True)
        shard_map, shard_size, quantized_count = _quantize_shard(
            source / shard_name,
            destination / shard_name,
            device,
        )
        weight_map.update(shard_map)
        total_size += shard_size
        total_quantized += quantized_count

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with output_index_path.open("w") as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        f"created={destination} shards={len(shard_names)} "
        f"quantized_weights={total_quantized} total_size={total_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
