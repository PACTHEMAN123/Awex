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

"""Two-rank CUDA smoke test for the experimental NCCL Device v2 backend."""

from __future__ import annotations  # noqa: I001

import os

# Import this module before torch so its configured NCCL preload wins SONAME
# resolution when the extension and PyTorch share one process.
from awex.transfer.nccl_device_v2 import _load_extension
import torch
import torch.distributed as dist

_MIB = 1024 * 1024
_TENSOR_BYTES = (65 * _MIB, 64 * _MIB, 32 * _MIB)
_PATTERNS = (0x11, 0x5A, 0xE3)


def _broadcast_unique_id(extension: object, rank: int) -> bytes:
    values = [extension.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("nccl_device_v2_smoke requires exactly two ranks")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    extension = _load_extension()
    unique_id = _broadcast_unique_id(extension, rank)
    tensors = [
        torch.full(
            (nbytes,),
            _PATTERNS[index] if rank == 0 else 0,
            dtype=torch.uint8,
            device="cuda",
        )
        for index, nbytes in enumerate(_TENSOR_BYTES)
    ]
    peer = 1 - rank
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
        metrics = extension.launch(
            handle,
            tensors,
            list(_TENSOR_BYTES),
            [0] * len(tensors),
            list(_TENSOR_BYTES),
            list(_TENSOR_BYTES),
            [peer] * len(tensors),
            list(range(len(tensors))),
            [len(tensors), 0] if rank == 1 else [0, len(tensors)],
            rank == 0,
            1,
        )
        dist.barrier()
        if rank == 1:
            for index, tensor in enumerate(tensors):
                if not torch.all(tensor == _PATTERNS[index]).item():
                    raise AssertionError(f"payload mismatch in tensor {index}")
        if metrics["topology_nvlink_count"] != 18:
            raise AssertionError(f"expected NV18 topology, got {dict(metrics)}")
        if metrics["topology_channels_per_peer"] != 64:
            raise AssertionError(f"expected 64 topology channels, got {dict(metrics)}")
        if metrics["channel_count"] != 32:
            raise AssertionError(f"expected 32 active channels, got {dict(metrics)}")
        if metrics["fragment_count"] <= metrics["work_count"]:
            raise AssertionError(
                f"expected a chunk crossing tensor spans, got {dict(metrics)}"
            )
        dist.barrier()
        print(f"rank={rank} metrics={dict(metrics)}", flush=True)
    finally:
        extension.destroy(handle)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
