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


import os
from importlib.metadata import version

import torch
import torch.distributed as dist
from packaging.version import Version

from shardstream import logging
from shardstream._utils import device as device_util

logger = logging.getLogger(__name__)


def init_custom_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size=-1,
    rank=-1,
    store=None,
    group_name=None,
    pg_options=None,
):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert store is None or init_method is None, (
        "Cannot specify both init_method and store."
    )
    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"
    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")
    if timeout is None:
        timeout = default_pg_timeout
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        (store, rank, world_size) = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)
    pg_options_param_name = (
        "backend_options"
        if Version(version("torch")) >= Version("2.6")
        else "pg_options"
    )
    (pg, _) = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg


def init_weights_update_group(
    master_address, master_port, rank, world_size, group_name, backend="nccl", role=""
):
    """Initialize the Torch process group for model parameter updates."""
    assert torch.distributed.is_initialized(), (
        "Default torch process group must be initialized"
    )
    assert group_name != "", "Group name cannot be empty"
    visible_env = device_util.visible_devices_env_value()
    logger.info(
        f"init custom process group for {role}: master_address={master_address}, master_port={master_port}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}, current device id {device_util.current_device()} {'/'.join(device_util.visible_devices_env_names())} {visible_env or '(unset)'} Local rank env {os.environ.get('LOCAL_RANK')} DEVICE env {os.environ.get('DEVICE')} Global rank env {os.environ.get('RANK')}"
    )
    try:
        options = None
        group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
            pg_options=options,
        )
        logger.info(f"Initialized custom process group: {group}")
        return group
    except Exception as e:
        raise RuntimeError(f"Failed to initialize custom process group: {e}.") from e


def setup_batch_isend_irecv(
    process_group, rank, world_size, tensor_size=10 * 10, dtype=torch.float32
):
    """
    Perform a simple communication using batch_isend_irecv to avoid the hang for later sub-ranks.

    Args:
    process_group (ProcessGroup): The process group to work on.
    tensor_size (int): Size of the tensor to send/receive.
    dtype (torch.dtype): Data type of the tensor.
    """
    assert process_group is not None, "Process group cannot be None"
    device = device_util.current_device()
    logger.info(
        f"Setup batch isend irecv for rank {rank} world size {world_size} device {device}"
    )
    torch_device = device_util.get_torch_device(device)
    send_tensor = torch.full(
        (tensor_size,), rank, dtype=dtype, device=torch_device, requires_grad=False
    )
    recv_tensor = torch.zeros(
        (tensor_size,), dtype=dtype, device=torch_device, requires_grad=False
    )
    ops = []
    mid_point = world_size // 2
    if world_size <= 1:
        logger.info(f"Skip batch isend/irecv setup because world size={world_size}.")
    elif world_size % 2 == 0:
        if rank < mid_point:
            target_rank = rank + mid_point
            if target_rank < world_size:
                ops.append(
                    dist.P2POp(
                        dist.irecv, recv_tensor, target_rank, group=process_group
                    )
                )
        else:
            target_rank = rank - mid_point
            if target_rank >= 0:
                ops.append(
                    dist.P2POp(
                        dist.isend, send_tensor, target_rank, group=process_group
                    )
                )
    else:
        recv_from = (rank - 1 + world_size) % world_size
        send_to = (rank + 1) % world_size
        ops.append(dist.P2POp(dist.irecv, recv_tensor, recv_from, group=process_group))
        ops.append(dist.P2POp(dist.isend, send_tensor, send_to, group=process_group))
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    device_util.synchronize(device_id=device_util.current_device())
    dist.barrier(group=process_group, device_ids=[device_util.current_device()])
    logger.info(
        f"Simple communication completed for process group of size {world_size}"
    )
    if world_size <= 1:
        return
    if world_size % 2 == 0:
        if rank < mid_point and rank + mid_point < world_size:
            expected_value = rank + mid_point
            assert torch.all(recv_tensor == expected_value), (
                f"Rank {rank} received incorrect data from rank {rank + mid_point}"
            )
    else:
        expected_value = (rank - 1 + world_size) % world_size
        assert torch.all(recv_tensor == expected_value), (
            f"Rank {rank} received incorrect data from rank {expected_value}"
        )
    logger.info("Simple communication verification successful")
