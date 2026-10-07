"""Exercise multiple ring roots sharing FIFO channels across four nodes."""

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from awex.tests.experimental import nccl_device_v2_multinode_e2e as single
from awex.transfer.nccl_device_v2 import NCCLDeviceV2Transport
from awex.transfer.transfer_plan import CommunicationOperation, TransferPlan


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    readers = 4 if os.environ.get("AWEX_RING_CHECK_COLLIDING_MIXED") == "1" else 2
    if world != readers + 10:
        raise ValueError(f"Use {readers} readers and 10 writers")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    parameters, operations = {}, {}
    for root in range(readers, world):
        if rank != root and rank >= readers:
            continue
        name, shape = f"model.root_{root}.weight", (16 * 1024 * 1024 + root * 64,)
        parameters[name] = torch.zeros(shape, dtype=torch.float32, device="cuda")
        shard = SimpleNamespace(name=name, shape=shape)
        for reader in range(readers):
            if rank not in (root, reader):
                continue
            peer = reader if rank == root else root
            operations.setdefault(peer, []).append(
                CommunicationOperation(
                    send_rank=root,
                    send_shard_meta=shard,
                    send_offset=(0,),
                    recv_rank=reader,
                    recv_shard_meta=shard,
                    recv_offset=(0,),
                    overlap_shape=shape,
                    train_slices=(slice(None),),
                    inf_slices=(slice(None),),
                )
            )
    plan = TransferPlan(operations=operations)
    transport = NCCLDeviceV2Transport(
        dist.group.WORLD,
        rank,
        world,
        timeout_ms=120_000,
        infer_instance_world_size=1,
        num_infer_engines=readers,
        ring_broadcast=True,
        ring_swizzle=True,
    )
    try:
        if rank >= readers:
            transport.prepare_send(parameters, plan, allow_staging=False)
        else:
            transport.prepare_recv(parameters, plan, allow_staging=False)
        for step in range(10):
            if rank >= readers:
                single._set_values(parameters, (rank + step / 16,))
            else:
                for tensor in parameters.values():
                    tensor.fill_(-999)
            dist.barrier(device_ids=[0])
            metrics = (transport.send if rank >= readers else transport.recv)(
                parameters, plan, step
            )
            dist.barrier(device_ids=[0])
            if rank < readers:
                for root in range(readers, world):
                    name = f"model.root_{root}.weight"
                    single._verify_values({name: parameters[name]}, (root + step / 16,))
                if metrics["ring_channel_collision_count"] == 0:
                    raise AssertionError("The test did not exercise channel collisions")
                if readers == 4 and metrics["lsa_peer_count"] == 0:
                    raise AssertionError("The mixed test did not exercise local relay")
            dist.barrier(device_ids=[0])
        print(
            "AWEX_V2_MULTINODE_E2E_PASS "
            + json.dumps(
                {
                    "rank": rank,
                    "updates": 10,
                    "readers": readers,
                    "lsa_peer_count": metrics["lsa_peer_count"],
                    "gin_peer_count": metrics["gin_peer_count"],
                    "ring_channel_collision_count": metrics[
                        "ring_channel_collision_count"
                    ],
                    "work_count": metrics["work_count"],
                }
            ),
            flush=True,
        )
    finally:
        transport.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
