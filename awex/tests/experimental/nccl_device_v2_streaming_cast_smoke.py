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

"""Two-rank mixed-format streaming-cast smoke test for device v2."""

from __future__ import annotations  # noqa: I001

import os

from awex.transfer.nccl_device_v2 import _load_extension
import torch
import torch.distributed as dist

_NUMELS = (5 * 1024 * 1024 + 257, 1024 * 1024 + 129)
_WIRE_DTYPES = (4, 1)  # FP8 E4M3 and FP16.
_WIRE_ELEMENT_BYTES = (1, 2)


def _broadcast_unique_id(extension: object, rank: int) -> bytes:
    values = [extension.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _values(sequence: int) -> list[torch.Tensor]:
    first = torch.arange(_NUMELS[0], dtype=torch.int32, device="cuda")
    second = torch.arange(_NUMELS[1], dtype=torch.int32, device="cuda")
    return [
        (((first + sequence * 37) % 4096) - 2048).to(torch.bfloat16) / 16,
        (((second + sequence * 19) % 8192) - 4096).float() / 32,
    ]


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("streaming-cast smoke test requires exactly two ranks")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("streaming-cast smoke test requires float8_e4m3fn")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    extension = _load_extension()
    unique_id = _broadcast_unique_id(extension, rank)
    peer = 1 - rank
    tensors = (
        [
            torch.empty(_NUMELS[0], dtype=torch.bfloat16, device="cuda"),
            torch.empty(_NUMELS[1], dtype=torch.float32, device="cuda"),
        ]
        if rank == 0
        else [
            torch.empty(_NUMELS[0], dtype=torch.float8_e4m3fn, device="cuda"),
            torch.empty(_NUMELS[1], dtype=torch.float16, device="cuda"),
        ]
    )
    handle = extension.create(
        unique_id,
        world_size,
        rank,
        local_rank,
        60_000,
        1,
        8,
        513,
        4096,
    )
    try:
        launches = []
        for sequence in (1, 2):
            values = _values(sequence)
            if rank == 0:
                for tensor, value in zip(tensors, values):
                    tensor.copy_(value)
            else:
                for tensor in tensors:
                    tensor.zero_()
            dist.barrier()
            metrics = extension.launch(
                handle,
                tensors,
                [
                    numel * wire_element_bytes
                    for numel, wire_element_bytes in zip(_NUMELS, _WIRE_ELEMENT_BYTES)
                ],
                [0, 0],
                [
                    numel * tensor.element_size()
                    for numel, tensor in zip(_NUMELS, tensors)
                ],
                [
                    numel * tensor.element_size()
                    for numel, tensor in zip(_NUMELS, tensors)
                ],
                list(_WIRE_DTYPES),
                list(_WIRE_ELEMENT_BYTES),
                [peer, peer],
                [0, 1],
                [2, 0] if rank == 1 else [0, 2],
                rank == 0,
                sequence,
            )
            launches.append(dict(metrics))
            dist.barrier()
            if rank == 1:
                expected = [
                    values[0].to(torch.float8_e4m3fn).float(),
                    values[1].to(torch.float16).float(),
                ]
                for index, (tensor, reference) in enumerate(zip(tensors, expected)):
                    if not torch.equal(tensor.float(), reference):
                        difference = (tensor.float() - reference).abs().max().item()
                        raise AssertionError(
                            f"streaming-cast tensor {index} mismatch: max_abs={difference}"
                        )

        metrics = launches[-1]
        expected_wire_bytes = sum(
            numel * itemsize for numel, itemsize in zip(_NUMELS, _WIRE_ELEMENT_BYTES)
        )
        expected_tensor_bytes = sum(
            numel * tensor.element_size() for numel, tensor in zip(_NUMELS, tensors)
        )
        if metrics["tensor_bytes"] != expected_tensor_bytes:
            raise AssertionError(f"unexpected tensor byte count: {metrics}")
        if metrics["wire_bytes"] != expected_wire_bytes:
            raise AssertionError(f"unexpected wire byte count: {metrics}")
        if metrics["streaming_cast_tasks"] != (2 if rank == 0 else 0):
            raise AssertionError(f"unexpected streaming-cast task count: {metrics}")
        if metrics["fragment_count"] <= metrics["work_count"]:
            raise AssertionError(f"no work crossed the tensor boundary: {metrics}")
        if not launches[1]["plan_cache_hit"]:
            raise AssertionError(f"second launch missed the plan cache: {metrics}")
        dist.barrier()
        print(f"rank={rank} streaming_cast={metrics}", flush=True)
    finally:
        extension.destroy(handle)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
