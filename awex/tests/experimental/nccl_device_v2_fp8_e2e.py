"""Changing BF16 -> native block FP8 correctness through real multi-node FIFO."""

# ruff: noqa: E402, I001
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_package = types.ModuleType("awex")
_package.__path__ = [str(Path(__file__).resolve().parents[2])]
sys.modules.setdefault("awex", _package)

import torch
import torch.distributed as dist
from awex.transfer.nccl_device_v2 import NCCLDeviceV2Transport
from awex.transfer.transfer_plan import CommunicationOperation, TransferPlan


def reference(source, br, bc):
    tiled = source.float().reshape(source.shape[0] // br, br, source.shape[1] // bc, bc)
    scale = tiled.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    # Match the kernel's reciprocal multiply and finite E4M3 conversion.
    quant = (
        (tiled * scale.reciprocal()[:, None, :, None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    return quant.reshape(source.shape), scale


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    writer, readers = world - 1, list(range(world - 1))
    br = int(os.environ.get("AWEX_NCCL_DEVICE_V2_FP8_BLOCK_ROWS", "128"))
    bc = int(os.environ.get("AWEX_NCCL_DEVICE_V2_FP8_BLOCK_COLS", "128"))
    shape = (4096, 1024)
    dtype = torch.bfloat16 if rank == writer else torch.float8_e4m3fn
    # Strided matrices test direct model writes without a staging tensor.
    storage = torch.zeros((shape[0], shape[1] + 128), dtype=dtype, device="cuda")
    tensor = storage[:, : shape[1]]
    name = "model.expert.weight"
    params = {name: tensor}
    if rank != writer:
        scale_storage = torch.full(
            (shape[0] // br, shape[1] // bc + 2), -17.0, device="cuda"
        )
        params[name + "_scale_inv"] = scale_storage[:, : shape[1] // bc]
    ops = {}
    for reader in readers:
        if rank not in (writer, reader):
            continue
        source = SimpleNamespace(name=name, shape=shape, dtype="bfloat16")
        target = SimpleNamespace(name=name, shape=shape, dtype="float8_e4m3fn")
        slices = (slice(None), slice(None))
        ops[reader if rank == writer else writer] = [
            CommunicationOperation(
                writer, source, (0, 0), reader, target, (0, 0), shape, slices, slices
            )
        ]
    transport = NCCLDeviceV2Transport(
        dist.group.WORLD,
        rank,
        world,
        timeout_ms=120000,
        infer_instance_world_size=1,
        num_infer_engines=len(readers),
    )
    plan = TransferPlan(operations=ops)
    if rank == writer:
        transport.prepare_send(params, plan, allow_staging=False)
    else:
        transport.prepare_recv(params, plan, allow_staging=False)
    metrics = []
    index = torch.arange(shape[0] * shape[1], device="cuda").reshape(shape)
    for step in range(1, 11):
        source = (((index * 13 + step * 11) % 1009).float() / 64 - 7.5).to(
            torch.bfloat16
        )
        if step == 1:
            source.zero_()
        if rank == writer:
            tensor.copy_(source)
        dist.barrier(device_ids=[torch.cuda.current_device()])
        result = (
            transport.send(params, plan, step)
            if rank == writer
            else transport.recv(params, plan, step)
        )
        metrics.append(result)
        if rank != writer:
            expected, scales = reference(source, br, bc)
            if not torch.equal(tensor.view(torch.uint8), expected.view(torch.uint8)):
                raise AssertionError(f"FP8 bytes differ: rank={rank}, step={step}")
            torch.testing.assert_close(
                params[name + "_scale_inv"], scales, rtol=1e-6, atol=0
            )
            assert torch.all(storage[:, shape[1] :].view(torch.uint8) == 0)
            assert torch.all(scale_storage[:, shape[1] // bc :] == -17)
        dist.barrier(device_ids=[torch.cuda.current_device()])
    transport.close()
    dist.destroy_process_group()
    print(
        "AWEX_V2_FP8_E2E_PASS "
        + json.dumps(
            {"rank": rank, "block": [br, bc], "updates": 10, "metrics": metrics}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
