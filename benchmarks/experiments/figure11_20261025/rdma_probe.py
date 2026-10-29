"""Two-host qualification using the unmodified official Awex transport."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from awex.meta.meta_server import MetaServerClient, start_meta_server, stop_meta_server
from awex.transfer.mooncake_comm import MooncakeTransport
from awex.transfer.transfer_plan import CommunicationOperation, TransferPlan
from shardstream.integrations.publication.awex_rdma import enable_rdma_audit

def main():
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=["send", "receive"])
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=29871)
    a = p.parse_args()
    torch.cuda.set_device(0)
    enable_rdma_audit()
    if a.role == "send":
        start_meta_server(host=a.host, port=a.port)
    transport = MooncakeTransport(MetaServerClient(a.host, a.port),
                                 1 if a.role == "send" else 0, timeout=120)
    op = CommunicationOperation(
        send_rank=1, recv_rank=0,
        send_shard_meta=SimpleNamespace(name="weight"),
        recv_shard_meta=SimpleNamespace(name="weight"),
        send_offset=(0, 1), recv_offset=(0, 1), overlap_shape=(1024, 1022),
        train_slices=(slice(None), slice(1, 1023)),
        inf_slices=(slice(None), slice(1, 1023)),
    )
    plan = TransferPlan(operations={0 if a.role == "send" else 1: [op]})
    try:
        for version in range(3):
            source = torch.arange(1024 * 1024, device="cuda", dtype=torch.float32).reshape(1024, 1024)
            source = (source % 97 + version).to(torch.bfloat16)
            target = torch.full((1024, 1024), -1., device="cuda", dtype=torch.bfloat16).T
            if a.role == "send":
                transport.send({"weight": source}, plan, step_id=42)
            else:
                transport.receive({"weight": target}, plan, step_id=42)
                assert torch.equal(target[:, 1:1023], source[:, 1:1023])
                assert torch.all(target[:, [0, 1023]] == -1)
            assert not transport.buffers
            print(json.dumps(dict(role=a.role, version=version, verified=True,
                                  protocol=transport.protocol, pid=os.getpid())), flush=True)
    finally:
        if a.role == "send":
            stop_meta_server()


if __name__ == "__main__":
    main()
