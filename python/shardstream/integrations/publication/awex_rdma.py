"""Attest the official Awex RDMA transport without changing its data path."""

import functools
import hashlib
import json
import os
from pathlib import Path


def enable_rdma_audit():
    if os.environ.get("AWEX_MOONCAKE_PROTOCOL") != "rdma":
        raise RuntimeError("Awex RDMA requires explicit AWEX_MOONCAKE_PROTOCOL=rdma")
    from awex.transfer import mooncake_comm

    cls = mooncake_comm.MooncakeTransport
    if getattr(cls, "_shardstream_audited", False):
        return
    source = Path(mooncake_comm.__file__)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    initialize = cls.__init__
    register = cls._register

    @functools.wraps(initialize)
    def audited_init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        if self.protocol != "rdma":
            raise RuntimeError(f"Awex transport selected {self.protocol}, expected rdma")
        print("AWEX_RDMA_TRANSPORT " + json.dumps({
            "pid": os.getpid(), "rank": self.rank, "protocol": self.protocol,
            "endpoint": self.endpoint, "module": str(source), "sha256": source_hash,
            "device": os.environ.get("AWEX_MOONCAKE_DEVICE", ""),
            "handshake_max_length": os.environ.get("MC_HANDSHAKE_MAX_LENGTH", ""),
        }), flush=True)

    @functools.wraps(register)
    def audited_register(self, tensor):
        if tensor.device.type != "cuda":
            raise RuntimeError(f"Awex RDMA benchmark requires CUDA buffers: {tensor.device}")
        return register(self, tensor)

    # Observe each completed native call. No tensor, plan, ordering or timing
    # boundary is changed; logging is once per role per process after success.
    def observe(method, role):
        @functools.wraps(method)
        def completed(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            flag = "_shardstream_observed_" + role
            if not getattr(self, flag, False):
                print("AWEX_RDMA_COMPLETE " + json.dumps({
                    "pid": os.getpid(), "rank": self.rank, "role": role,
                    "protocol": self.protocol,
                }), flush=True)
                setattr(self, flag, True)
            return result
        return completed

    cls.__init__ = audited_init
    cls._register = audited_register
    cls.send = observe(cls.send, "send")
    cls.receive = observe(cls.receive, "receive")
    cls._shardstream_audited = True
