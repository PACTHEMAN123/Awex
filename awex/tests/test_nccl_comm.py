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

import torch.distributed as dist

from awex.transfer.nccl_comm import _group_homogeneous_p2p_ops_by_ring_stage


def _op(fn, peer):
    return SimpleNamespace(op=fn, peer=peer)


def test_ring_stages_match_sender_and_receiver():
    send_ops = [_op(dist.isend, 5), _op(dist.isend, 6), _op(dist.isend, 5)]
    recv_ops = [_op(dist.irecv, 2), _op(dist.irecv, 2)]

    send_stages = _group_homogeneous_p2p_ops_by_ring_stage(
        send_ops, rank=2, world_size=8
    )
    recv_stages = _group_homogeneous_p2p_ops_by_ring_stage(
        recv_ops, rank=5, world_size=8
    )

    assert list(send_stages) == [3, 4]
    assert [op.peer for op in send_stages[3]] == [5, 5]
    assert list(recv_stages) == [3]
    assert len(recv_stages[3]) == 2


def test_ring_stages_reject_mixed_direction_batch():
    ops = [_op(dist.isend, 5), _op(dist.irecv, 1)]

    assert (
        _group_homogeneous_p2p_ops_by_ring_stage(ops, rank=2, world_size=8)
        is None
    )
