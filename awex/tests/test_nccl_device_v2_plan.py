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

from awex.transfer import nccl_device_v2
from awex.transfer.nccl_device_v2 import (
    NCCLDeviceV2UnavailableError,
    _build_recv_batch,
    _build_send_batch,
)
from awex.transfer.transfer_plan import CommunicationOperation, TransferPlan


def _operation(source_dtype, destination_dtype, numel=32):
    send_shard = SimpleNamespace(name="weight", shape=(numel,), dtype=source_dtype)
    recv_shard = SimpleNamespace(name="weight", shape=(numel,), dtype=destination_dtype)
    return CommunicationOperation(
        send_rank=1,
        send_shard_meta=send_shard,
        send_offset=(0,),
        recv_rank=0,
        recv_shard_meta=recv_shard,
        recv_offset=(0,),
        overlap_shape=(numel,),
        train_slices=(slice(None),),
        inf_slices=(slice(None),),
    )


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="PyTorch has no FP8 dtype"
)
def test_streaming_cast_uses_destination_dtype_as_wire_format(monkeypatch):
    operation = _operation(torch.bfloat16, torch.float8_e4m3fn)
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)

    send_batch = _build_send_batch(
        {"weight": torch.arange(32, dtype=torch.bfloat16)},
        TransferPlan(operations={0: [operation]}),
        rank=1,
        world_size=2,
        chunk_bytes=16,
        allow_staging=False,
    )
    recv_batch = _build_recv_batch(
        {"weight": torch.empty(32, dtype=torch.float8_e4m3fn)},
        TransferPlan(operations={1: [operation]}),
        rank=0,
        world_size=2,
        chunk_bytes=16,
        allow_staging=False,
    )

    assert send_batch.lengths == recv_batch.lengths == [32]
    assert send_batch.tensor_lengths == [64]
    assert recv_batch.tensor_lengths == [32]
    assert send_batch.wire_dtypes == recv_batch.wire_dtypes == [4]
    assert send_batch.wire_element_bytes == [1]
    assert send_batch.expected_counts == [1, 0]
    assert recv_batch.expected_counts == [0, 1]


def test_streaming_cast_rejects_unsupported_numeric_conversion(monkeypatch):
    operation = _operation(torch.int32, torch.float32)
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)

    with pytest.raises(
        NCCLDeviceV2UnavailableError, match="streaming cast does not support"
    ):
        _build_send_batch(
            {"weight": torch.arange(32, dtype=torch.int32)},
            TransferPlan(operations={0: [operation]}),
            rank=1,
            world_size=2,
            chunk_bytes=16,
            allow_staging=False,
        )


def test_receive_tensor_must_match_wire_dtype(monkeypatch):
    operation = _operation(torch.bfloat16, torch.float16)
    monkeypatch.setattr(nccl_device_v2, "_ensure_cuda_tensor", lambda *_: None)

    with pytest.raises(
        NCCLDeviceV2UnavailableError, match="does not match the wire format"
    ):
        _build_recv_batch(
            {"weight": torch.empty(32, dtype=torch.bfloat16)},
            TransferPlan(operations={1: [operation]}),
            rank=0,
            world_size=2,
            chunk_bytes=16,
            allow_staging=False,
        )
