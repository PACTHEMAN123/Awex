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


@pytest.mark.parametrize("bias_suffix", ["down_proj_bias", "down_proj.bias", "w2_bias"])
def test_gpt_oss_loader_cannot_zero_bias_in_forwarded_native_bucket(monkeypatch, bias_suffix):
    import asyncio
    import threading

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    bucket = torch.arange(12, dtype=torch.bfloat16)
    original = bucket.clone()
    names = ["model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.mlp.experts." + bias_suffix,
             "model.layers.0.self_attn.k_proj.weight"]
    forwarded = []

    async def receive_weights(global_steps):
        assert global_steps == 3
        for index, name in enumerate(names):
            yield name, bucket[index * 4:(index + 1) * 4]
            # A downstream rank must see the original bucket after this
            # receiver's loader has run, irrespective of copy timing.
            forwarded.append(bucket.clone())
            assert torch.equal(bucket, original)

    loaded = {}

    def load_weights(weights):
        for index, (name, weight) in enumerate(weights):
            if name == names[1]:
                assert weight.data_ptr() != bucket[4:8].data_ptr()
                weight.zero_()  # Real vLLM GPT-OSS nonzero-TP loader behavior.
            else:
                assert weight.data_ptr() == bucket[index * 4:].data_ptr()
            loaded[name] = weight.clone()

    receiver = NativeReceiver({"timeout_seconds": 5}, 1)
    receiver.engine = SimpleNamespace(receive_weights=receive_weights)
    receiver.loop = asyncio.new_event_loop()
    thread = threading.Thread(target=receiver.loop.run_forever)
    thread.start()
    model = SimpleNamespace(config=SimpleNamespace(architectures=["GptOssForCausalLM"]),
                            load_weights=load_weights)
    try:
        metrics = receiver.update(model, step_id=3)
        assert metrics["received_tensors"] == 3
        assert metrics["payload_bytes"] == bucket.numel() * bucket.element_size()
        assert len(forwarded) == 3
        assert loaded[names[1]].count_nonzero() == 0
        assert torch.equal(bucket, original)
    finally:
        receiver.loop.call_soon_threadsafe(receiver.loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive()
        receiver.loop.close()
