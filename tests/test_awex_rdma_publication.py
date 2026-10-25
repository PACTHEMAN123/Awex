from types import SimpleNamespace

import pytest

from shardstream.integrations.publication.awex_native import AwexRDMAPublication
from shardstream.integrations.publication.awex_rdma import enable_rdma_audit


@pytest.mark.parametrize("protocol", [None, "tcp"])
def test_rdma_refuses_implicit_or_tcp_protocol(monkeypatch, protocol):
    monkeypatch.delenv("AWEX_MOONCAKE_PROTOCOL", raising=False)
    if protocol is not None:
        monkeypatch.setenv("AWEX_MOONCAKE_PROTOCOL", protocol)
    with pytest.raises(RuntimeError, match="explicit"):
        enable_rdma_audit()


def test_rdma_requests_upstream_mooncake_reader(monkeypatch):
    from shardstream.integrations.publication import awex_native

    requests = []
    monkeypatch.setattr(awex_native.requests, "post", lambda url, **kwargs:
                        requests.append((url, kwargs)) or SimpleNamespace(status_code=200))
    harness = SimpleNamespace(meta_server_addr="host:1234", inference_config={"num_engines": 2})
    publication = AwexRDMAPublication(harness)
    publication._initialize_endpoint((1, "receiver", 4321))
    assert requests[0][0] == "http://receiver:4321/areal_awex_init"
    assert requests[0][1]["json"]["comm_backend"] == "mooncake"
    assert requests[0][1]["json"]["num_engines"] == 2
