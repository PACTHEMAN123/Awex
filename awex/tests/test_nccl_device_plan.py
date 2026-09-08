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

from types import SimpleNamespace

import pytest
import torch

from awex.transfer import nccl_device
from awex.transfer.nccl_device import (
    NCCLDeviceUnavailableError,
    _build_send_batch,
    _resolve_chunk_bytes,
    _split_contiguous_tensor,
)
from awex.transfer.transfer_plan import (
    CommunicationOperation,
    TransferChunk,
    TransferPlan,
    build_transfer_chunks,
)


def test_build_transfer_chunks_uses_fixed_ranges_and_short_tail():
    assert build_transfer_chunks(10, 4) == [
        TransferChunk(0, 4),
        TransferChunk(4, 4),
        TransferChunk(8, 2),
    ]


def test_build_transfer_chunks_can_preserve_one_task_per_tensor():
    assert build_transfer_chunks(10, 0) == [TransferChunk(0, 10)]
    assert build_transfer_chunks(0, 4) == []


def test_split_contiguous_tensor_preserves_storage_and_byte_ranges():
    tensor = torch.arange(10, dtype=torch.int32)

    chunks = _split_contiguous_tensor(tensor, 16)

    assert [(offset, length) for _, offset, length in chunks] == [
        (0, 16),
        (16, 16),
        (32, 8),
    ]
    assert torch.equal(torch.cat([chunk for chunk, _, _ in chunks]), tensor)
    assert chunks[1][0].data_ptr() == tensor.data_ptr() + 16


def test_chunk_size_must_be_vector_aligned():
    with pytest.raises(NCCLDeviceUnavailableError, match="multiple of 16"):
        _resolve_chunk_bytes(15)


def test_chunk_size_rejects_negative_values():
    with pytest.raises(NCCLDeviceUnavailableError, match="non-negative"):
        _resolve_chunk_bytes(-16)


def test_send_plan_lowers_tensor_to_chunk_tasks(monkeypatch):
    tensor = torch.arange(10, dtype=torch.int32)
    shard = SimpleNamespace(name="weight")
    operation = CommunicationOperation(
        send_rank=1,
        send_shard_meta=shard,
        send_offset=(0,),
        recv_rank=0,
        recv_shard_meta=shard,
        recv_offset=(0,),
        overlap_shape=(10,),
        train_slices=(slice(None),),
        inf_slices=(slice(None),),
    )
    plan = TransferPlan(operations={0: [operation]})
    monkeypatch.setattr(nccl_device, "_ensure_cuda_tensor", lambda *_: None)

    batch = _build_send_batch(
        {"weight": tensor}, plan, rank=1, world_size=2, chunk_bytes=16
    )

    assert batch.offsets == [0, 16, 32]
    assert batch.lengths == [16, 16, 8]
    assert batch.ordinals == [0, 1, 2]
    assert batch.expected_counts == [3, 0]
    assert batch.region_bytes == [0, 40]
    assert torch.equal(torch.cat(batch.tensors), tensor)
