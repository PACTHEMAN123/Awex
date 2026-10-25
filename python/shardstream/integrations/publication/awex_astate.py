"""Configure native AState discovery and record its deployed artifact identity."""
import functools
import hashlib
import json
import os
from pathlib import Path


def configure_astate_discovery(meta_server_addr):
    if "UTRANS_DISABLE_RDMA" in os.environ:
        raise RuntimeError("AState qualification requires native RDMA enabled")
    if os.environ.get("TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION", "false").lower() != "false":
        raise RuntimeError("AState qualification cannot ignore RDMA failures")
    host, port = meta_server_addr.rsplit(":", 1)
    os.environ.update(DISCOVERY_CONFIG_CENTER_TYPE="HTTP",
                      ASYSTEM_META_SERVER_IP=host, ASYSTEM_META_SERVER_PORT=port,
                      TRANSFER_ENGINE_SKIP_RDMA_EXCEPTION="false")


def enable_astate_audit():
    from awex.reader.astate_reader import AStateWorkerWeightsReader
    from awex.writer.astate_writer import AStateWeightsWriter

    for cls, method_name, role in [(AStateWeightsWriter, "_initialize", "send"),
                                   (AStateWorkerWeightsReader, "initialize", "receive")]:
        if getattr(cls, "_shardstream_astate_audited", False):
            continue

        def observe_initialize(method, role):
            @functools.wraps(method)
            def initialize(self, *args, **kwargs):
                configure_astate_discovery(self.meta_server_addr)
                result = method(self, *args, **kwargs)
                manifest_path = Path(os.environ["SHARDSTREAM_ASTATE_MANIFEST"])
                manifest = json.loads(manifest_path.read_text())
                import astate._core.astate_cpp as native
                binary = Path(native.__file__)
                binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
                if binary_hash != manifest["extension_sha256"]:
                    raise RuntimeError("AState extension differs from the qualified artifact")
                library = Path(manifest["utrans_library"])
                if hashlib.sha256(library.read_bytes()).hexdigest() != manifest["utrans_sha256"]:
                    raise RuntimeError("AState libutrans differs from the qualified artifact")
                print("AWEX_ASTATE_TRANSPORT " + json.dumps(dict(
                    pid=os.getpid(), rank=self.transfer_rank, role=role, protocol="rdma",
                    module=str(binary), sha256=binary_hash,
                    source_revision=manifest["source_revision"],
                    utrans_sha256=manifest["utrans_sha256"], staging="cpu_pinned",
                    max_rdma_devices=os.environ.get("TRANSFER_ENGINE_MAX_RDMA_DEVICES", "2"),
                    discovery="HTTP")), flush=True)
                return result
            return initialize

        def observe_complete(method, role):
            @functools.wraps(method)
            def complete(self, *args, **kwargs):
                result = method(self, *args, **kwargs)
                if not getattr(self, "_shardstream_astate_completed", False):
                    print("AWEX_ASTATE_COMPLETE " + json.dumps(dict(
                        pid=os.getpid(), rank=self.transfer_rank, role=role,
                        protocol="rdma")), flush=True)
                    self._shardstream_astate_completed = True
                return result
            return complete

        setattr(cls, method_name, observe_initialize(getattr(cls, method_name), role))
        cls.finish_step = observe_complete(cls.finish_step, role)
        cls._shardstream_astate_audited = True
