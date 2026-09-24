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

"""Two-rank BF16-to-FP8 streaming-cast smoke test for device v2."""

from __future__ import annotations  # noqa: I001

import os

from awex.transfer.nccl_device_v2 import _load_extension
import torch
import torch.distributed as dist

_NUMEL = 5 * 1024 * 1024 + 257
_FP8_E4M3_CODE = 4


def _broadcast_unique_id(extension: object, rank: int) -> bytes:
    values = [extension.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _values(sequence: int) -> torch.Tensor:
    indices = torch.arange(_NUMEL, dtype=torch.int32, device="cuda")
    return (((indices + sequence * 37) % 4096) - 2048).to(torch.bfloat16) / 16


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
    tensor = (
        torch.empty(_NUMEL, dtype=torch.bfloat16, device="cuda")
        if rank == 0
        else torch.empty(_NUMEL, dtype=torch.float8_e4m3fn, device="cuda")
    )
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
            values = _values(sequence)
            if rank == 0:
                tensor.copy_(values)
            else:
                tensor.zero_()
            dist.barrier()
            metrics = extension.launch(
                handle,
                [tensor],
                [_NUMEL],
                [0],
                [_NUMEL * tensor.element_size()],
                [_NUMEL * tensor.element_size()],
                [_FP8_E4M3_CODE],
                [1],
                [peer],
                [0],
                [1, 0] if rank == 1 else [0, 1],
                rank == 0,
                sequence,
            )
            launches.append(dict(metrics))
            dist.barrier()
            if rank == 1:
                expected = values.to(torch.float8_e4m3fn).float()
                if not torch.equal(tensor.float(), expected):
                    difference = (tensor.float() - expected).abs().max().item()
                    raise AssertionError(f"BF16-to-FP8 mismatch: max_abs={difference}")

        metrics = launches[-1]
        expected_tensor_bytes = _NUMEL * (2 if rank == 0 else 1)
        if metrics["tensor_bytes"] != expected_tensor_bytes:
            raise AssertionError(f"unexpected tensor byte count: {metrics}")
        if metrics["wire_bytes"] != _NUMEL:
            raise AssertionError(f"unexpected wire byte count: {metrics}")
        if metrics["streaming_cast_tasks"] != (1 if rank == 0 else 0):
            raise AssertionError(f"unexpected streaming-cast task count: {metrics}")
        if not launches[1]["plan_cache_hit"]:
            raise AssertionError(f"second launch missed the plan cache: {metrics}")
        dist.barrier()
        print(f"rank={rank} streaming_cast={metrics}", flush=True)
    finally:
        extension.destroy(handle)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
