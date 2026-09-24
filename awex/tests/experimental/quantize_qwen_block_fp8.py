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
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from awex.converter.weights_converter import per_block_cast_to_fp8

_ATTENTION_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
)
_MLP_PROJECTION_SUFFIXES = (
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)
_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Create shape-correct dummy tensors without reading source tensor data.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _should_quantize(name: str, shape: tuple[int, ...]) -> bool:
    is_attention_weight = name.endswith(_ATTENTION_SUFFIXES)
    is_mlp_projection = ".mlp." in name and name.endswith(
        _MLP_PROJECTION_SUFFIXES
    )
    is_router_weight = name.endswith(".mlp.gate.weight")
    return len(shape) == 2 and (
        is_attention_weight or is_mlp_projection or is_router_weight
    )


def _torch_dtype(safetensors_dtype) -> torch.dtype:
    dtype_name = str(safetensors_dtype).rsplit(".", 1)[-1]
    try:
        return _SAFETENSORS_DTYPES[dtype_name]
    except KeyError as error:
        raise ValueError(f"Unsupported safetensors dtype: {dtype_name}") from error


def _copy_auxiliary_files(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".safetensors" or path.name in {
            "config.json",
            "model.safetensors.index.json",
        }:
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
    staging_path: Path,
    device: torch.device,
    dummy: bool,
) -> tuple[dict[str, str], int, int]:
    tensors: dict[str, torch.Tensor] = {}
    weight_map = {}
    quantized_count = 0
    total_size = 0
    with safe_open(source_path, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        for name in source.keys():
            tensor_slice = source.get_slice(name)
            shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            if _should_quantize(name, shape):
                if dummy:
                    tensors[name] = torch.zeros(shape, dtype=torch.float8_e4m3fn)
                    scale = torch.ones(
                        ((shape[0] + 127) // 128, (shape[1] + 127) // 128),
                        dtype=torch.float32,
                    )
                else:
                    tensor = source.get_tensor(name)
                    quantized, scale = per_block_cast_to_fp8(
                        tensor.to(device=device, non_blocking=False), False
                    )
                    tensors[name] = quantized.cpu()
                    scale = scale.cpu()
                scale_name = f"{name}_scale_inv"
                tensors[scale_name] = scale
                weight_map[scale_name] = destination_path.name
                quantized_count += 1
            else:
                tensors[name] = (
                    torch.zeros(shape, dtype=_torch_dtype(tensor_slice.get_dtype()))
                    if dummy
                    else source.get_tensor(name)
                )
            weight_map[name] = destination_path.name

    for tensor in tensors.values():
        total_size += int(tensor.numel()) * int(tensor.element_size())
    save_file(tensors, staging_path, metadata=metadata)
    staging_size = staging_path.stat().st_size
    try:
        staging_path.replace(destination_path)
    except OSError:
        shutil.copyfile(staging_path, destination_path)
        staging_path.unlink()
    if destination_path.stat().st_size != staging_size:
        raise OSError(f"Copied shard size mismatch: {destination_path}")
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
    for incomplete_path in destination.glob("*.incomplete"):
        incomplete_path.unlink()
    _copy_auxiliary_files(source, destination)
    _write_config(source, destination)

    device = torch.device(args.device)
    weight_map = {}
    total_size = 0
    total_quantized = 0
    with tempfile.TemporaryDirectory(prefix="awex-qwen-block-fp8-") as staging_dir:
        staging = Path(staging_dir)
        for index, shard_name in enumerate(shard_names, start=1):
            print(
                f"[{index}/{len(shard_names)}] quantizing {shard_name}", flush=True
            )
            shard_map, shard_size, quantized_count = _quantize_shard(
                source / shard_name,
                destination / shard_name,
                staging / shard_name,
                device,
                args.dummy,
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
