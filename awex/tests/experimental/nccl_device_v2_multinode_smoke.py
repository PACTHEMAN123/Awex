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

"""Two-node smoke test for the GIN path inside nccl_device_v2."""

from __future__ import annotations  # noqa: I001

import os

from awex.transfer.nccl_device_v2 import _load_extension
import torch
import torch.distributed as dist

_MIB = 1024 * 1024
_TENSOR_BYTES = (8 * _MIB, 5 * _MIB)
_PATTERNS = (0x2D, 0xC7)


def _broadcast_unique_id(extension: object, rank: int) -> bytes:
    values = [extension.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("nccl_device_v2_multinode_smoke requires two ranks")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    extension = _load_extension()
    unique_id = _broadcast_unique_id(extension, rank)
    peer = 1 - rank
    tensors = [
        torch.full(
            (nbytes,),
            _PATTERNS[index] if rank == 0 else 0,
            dtype=torch.uint8,
            device="cuda",
        )
        for index, nbytes in enumerate(_TENSOR_BYTES)
    ]
    handle = extension.create(
        unique_id,
        world_size,
        rank,
        local_rank,
        120_000,
        64,
        8,
        512 * 1024,
        128 * 1024,
        4 * 1024 * 1024,
        0,
    )
    try:
        launch_metrics = []
        for sequence in (1, 2):
            if rank == 1 and sequence > 1:
                for tensor in tensors:
                    tensor.zero_()
            dist.barrier()
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
                sequence,
            )
            launch_metrics.append(metrics)
            dist.barrier()
            if rank == 1:
                for index, tensor in enumerate(tensors):
                    if not torch.all(tensor == _PATTERNS[index]).item():
                        raise AssertionError(
                            f"payload mismatch in tensor {index} at sequence {sequence}"
                        )

        metrics = launch_metrics[-1]
        if metrics["lsa_peer_count"] != 0 or metrics["gin_peer_count"] != 1:
            raise AssertionError(f"expected the GIN path, got {dict(metrics)}")
        if not metrics["gin_enabled"] or metrics["gin_context_count"] <= 0:
            raise AssertionError(f"GIN was not initialized: {dict(metrics)}")
        if metrics["gin_connection_count"] <= 0:
            raise AssertionError(f"GIN has no network connections: {dict(metrics)}")
        if metrics["network_step_bytes"] != 128 * 1024:
            raise AssertionError(
                f"expected NCCL network step size, got {dict(metrics)}"
            )
        if metrics["min_work_step_bytes"] != 128 * 1024:
            raise AssertionError(f"expected 128 KiB GIN steps, got {dict(metrics)}")
        if metrics["fifo_depth"] != 8 or metrics["gin_fifo_depth"] != 16:
            raise AssertionError(
                f"LSA and GIN FIFO settings were not isolated: {dict(metrics)}"
            )
        if metrics["chunk_bytes"] != 4 * 1024 * 1024:
            raise AssertionError(f"main's LSA chunk size changed: {dict(metrics)}")
        if metrics["gin_chunk_bytes"] != 4 * 1024 * 1024:
            raise AssertionError(f"unexpected GIN chunk size: {dict(metrics)}")
        if metrics["channel_count"] != metrics["network_channels_per_peer"]:
            raise AssertionError(
                f"GIN did not use its network channels: {dict(metrics)}"
            )
        if metrics["payload_peer_count"] != 1:
            raise AssertionError(f"expected one payload peer, got {dict(metrics)}")
        if launch_metrics[0]["plan_cache_hit"]:
            raise AssertionError(
                f"first launch unexpectedly hit cache: {dict(metrics)}"
            )
        if not launch_metrics[1]["plan_cache_hit"]:
            raise AssertionError(f"second launch missed cache: {dict(metrics)}")
        dist.barrier()
        print(
            f"rank={rank} first={dict(launch_metrics[0])} "
            f"cached={dict(launch_metrics[1])}",
            flush=True,
        )
    finally:
        extension.destroy(handle)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
