from types import SimpleNamespace

import pytest
import torch

from shardstream.integrations.publication.verl_native import NativePublication, NativeReceiver


@pytest.mark.parametrize("coverage_ok", [True, False])
def test_mooncake_stale_ack_is_cleared_only_after_all_receivers_finish(monkeypatch, coverage_ok):
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    harness = SimpleNamespace(
        is_driver=True,
        megatron_engine=SimpleNamespace(global_step=0),
        inference_config={"tp_size": 2},
        publication_endpoints=lambda: [(0, "host0", 1), (1, "host1", 2)],
        _training_barrier=lambda: events.append("barrier"),
    )
    publisher = NativePublication(harness)
    publisher.backend = "mooncake"
    publisher._export = lambda: iter([("weight", torch.ones(2, dtype=torch.bfloat16))])
    acknowledgements = torch.full((8,), 0xAB, dtype=torch.uint8)

    async def send(weights, global_steps):
        assert len(list(weights)) == 1
        events.append("source_done")

    publisher.engine = SimpleNamespace(send_weights=send, magic_recv=acknowledgements)

    class PendingReceivers:
        def result(self):
            assert "source_done" in events
            assert acknowledgements.count_nonzero() == 8
            events.append("all_receivers_done")
            return [{"received_tensors": 1, "payload_bytes": 4}] * (4 if coverage_ok else 3)

    publisher.control = SimpleNamespace(submit=lambda *args: PendingReceivers())

    def reset_receivers(route, payloads):
        assert "all_receivers_done" in events
        assert acknowledgements.count_nonzero() == 0
        assert route == "publication_configure"
        events.append("reset_receivers")
        results = []
        for engine in range(2):
            for worker in range(2):
                receiver = NativeReceiver({}, worker)
                receiver.backend = "mooncake"
                stale = torch.full((8,), 0xAB, dtype=torch.uint8)
                receiver.engine = SimpleNamespace(magic_recv=stale)
                results.append(receiver.configure(payloads[engine]["config"]))
                assert stale.count_nonzero() == 0
        return results

    publisher._requests = reset_receivers
    if coverage_ok:
        publisher.publish()
        assert events.index("all_receivers_done") < events.index("reset_receivers")
        assert publisher.last_metrics["completion_reset_ms"] >= 0
    else:
        with pytest.raises(RuntimeError, match="coverage"):
            publisher.publish()
        assert "reset_receivers" not in events
        assert acknowledgements.count_nonzero() == 8


def test_completion_reset_rejects_other_backends():
    receiver = NativeReceiver({}, 0)
    receiver.backend = "nixl"
    with pytest.raises(ValueError, match="only applies to Mooncake"):
        receiver.configure({"reset_completion": True})
