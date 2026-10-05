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

"""Cross-node CUDA smoke test for the experimental NCCL Device v2 backend."""

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
    if world_size not in (2, 4):
        raise RuntimeError("nccl_device_v2_smoke requires two or four ranks")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    extension = _load_extension()
    unique_id = _broadcast_unique_id(extension, rank)
    sender = rank < world_size // 2
    peer = world_size - 1 - rank
    patterns = tuple((pattern + (rank if sender else peer)) & 0xFF for pattern in _PATTERNS)
    tensors = [
        torch.full(
            (nbytes,),
            patterns[index] if sender else 0,
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
        60_000,
        int(os.environ.get("AWEX_NCCL_DEVICE_V2_MAX_CHANNELS", "64")),
        8,
        512 * 1024,
        4 * 1024 * 1024,
    )
    try:
        launch_metrics = []
        for sequence in (1, 2):
            if not sender and sequence > 1:
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
                [len(tensors) if index == peer else 0 for index in range(world_size)],
                sender,
                sequence,
            )
            launch_metrics.append(metrics)
            dist.barrier()
            if not sender:
                for index, tensor in enumerate(tensors):
                    if not torch.all(tensor == patterns[index]).item():
                        raise AssertionError(
                            f"payload mismatch in tensor {index} at sequence {sequence}"
                        )

        metrics = launch_metrics[-1]
        transport = metrics["transport"]
        if transport == "lsa" and metrics["topology_nvlink_count"] != 18:
            raise AssertionError(f"expected NV18 topology, got {dict(metrics)}")
        if transport == "lsa" and metrics["topology_raw_channels"] != 36:
            raise AssertionError(
                f"expected 36 raw topology channels, got {dict(metrics)}"
            )
        if metrics["topology_requested_channels_per_peer"] != 64:
            raise AssertionError(f"expected 64 requested channels, got {dict(metrics)}")
        if transport == "lsa" and metrics["topology_channels_per_peer"] != 64:
            raise AssertionError(
                f"expected 64 effective topology channels, got {dict(metrics)}"
            )
        if transport == "lsa" and metrics["channel_count"] != 32:
            raise AssertionError(
                f"expected 32 active channels, got {dict(metrics)}"
            )
        if transport != "lsa" and not 0 < metrics["channel_count"] <= metrics["channel_limit"]:
            raise AssertionError(f"invalid GIN channel count, got {dict(metrics)}")
        if metrics["threads_per_channel"] != 640:
            raise AssertionError(f"expected 640 channel threads, got {dict(metrics)}")
        if metrics["warps_per_channel"] != 20:
            raise AssertionError(f"expected 20 channel warps, got {dict(metrics)}")
        expected_payload_peers = (1 if sender else 0) if transport == "lsa" else world_size
        if metrics["payload_peer_count"] != expected_payload_peers:
            raise AssertionError(
                f"expected {expected_payload_peers} payload peers, got {dict(metrics)}"
            )
        if transport == "lsa" and metrics["registered_window_bytes"] >= metrics["dense_window_bytes"]:
            raise AssertionError(f"expected a sparse window, got {dict(metrics)}")
        if transport == "gin" and metrics["registered_window_bytes"] != metrics["dense_window_bytes"]:
            raise AssertionError(f"expected a dense GIN window, got {dict(metrics)}")
        if launch_metrics[0]["plan_cache_hit"]:
            raise AssertionError(f"first launch unexpectedly hit cache: {dict(metrics)}")
        if not launch_metrics[1]["plan_cache_hit"]:
            raise AssertionError(f"second launch missed cache: {dict(metrics)}")
        if launch_metrics[1]["host_lowering_time_ms"] != 0.0:
            raise AssertionError(f"cached launch repeated lowering: {dict(metrics)}")
        if launch_metrics[1]["metadata_upload_time_ms"] != 0.0:
            raise AssertionError(f"cached launch repeated metadata upload: {dict(metrics)}")
        if metrics["fragment_count"] <= metrics["work_count"]:
            raise AssertionError(
                f"expected a chunk crossing tensor spans, got {dict(metrics)}"
            )
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
