"""Validate native veRL's real multi-bucket transport with ten changing payloads."""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from awex.publication.verl_native_nccl import NativeVerlReceiver, NativeVerlSender


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = int(os.environ.get("AWEX_NATIVE_TEST_DEVICE", "0"))
    torch.cuda.set_device(device)
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        int(os.environ["MASTER_PORT"]),
        world,
        rank == 0,
        timeout=timedelta(seconds=180),
    )
    group_id = "native-verl-payload-" + os.environ["MASTER_PORT"]
    bucket_size = 256 << 10
    sender = receiver = None
    if rank == 0:
        sender = NativeVerlSender(
            world_size=world, group_id=group_id, bucket_size=bucket_size
        )
        store.set(
            "metadata",
            json.dumps(
                {
                    "store_host": sender.host,
                    "store_port": sender.port,
                    "world_size": world,
                    "group_id": group_id,
                    "bucket_size": bucket_size,
                }
            ),
        )
        sender.initialize_process_group()
    else:
        config = json.loads(store.get("metadata"))
        receiver = NativeVerlReceiver(config, worker_rank=rank - 1)
        receiver.initialize()

    def weights(step):
        for name, count in (("split", 524289), ("small", 51), ("tail", 4)):
            yield (
                name,
                ((torch.arange(count, device="cuda") + step * 31) % 997).to(
                    torch.bfloat16
                ),
            )

    class Model:
        def load_weights(self, iterator):
            for name, tensor in iterator:
                expected = dict(weights(self.step))[name]
                if not torch.equal(tensor, expected):
                    raise AssertionError(f"rank={rank} step={self.step} tensor={name}")
                self.loaded.add(name)

    model = Model()
    try:
        for step in range(10):
            if rank == 0:
                result = sender.broadcast_weights(step, weights(step))
                if result["tensor_count"] != 3 or result["bucket_count"] < 4:
                    raise AssertionError(result)
            else:
                model.step, model.loaded = step, set()
                result = receiver.update(model, step)
                if model.loaded != {"split", "small", "tail"}:
                    raise AssertionError(model.loaded)
            if result["payload_bytes"] != (524289 + 51 + 4) * 2:
                raise AssertionError(result)
            store.set(f"done/{step}/{rank}", "1")
            store.wait([f"done/{step}/{peer}" for peer in range(world)])
        print(
            f"NATIVE_VERL_NCCL_PAYLOAD_PASS rank={rank} device={device} updates=10",
            flush=True,
        )
    finally:
        if sender is not None:
            sender.close()
        if receiver is not None:
            receiver.close()


if __name__ == "__main__":
    main()
