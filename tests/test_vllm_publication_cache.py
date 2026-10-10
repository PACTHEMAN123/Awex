from types import SimpleNamespace

import pytest
from shardstream.integrations.vllm.server import ShardStreamVLLMServerAdapter


def adapter_with_cache(events, reset_ok=True):
    adapter = ShardStreamVLLMServerAdapter.__new__(ShardStreamVLLMServerAdapter)
    adapter._initialized = True
    adapter.weights_exchange_reader = SimpleNamespace(
        update_weights=lambda **kwargs: events.append("publish")
    )
    adapter._engine_client = SimpleNamespace(reset_encoder_cache=lambda: None)

    def call(method, *args):
        events.append(method)
        return reset_ok if method == "reset_prefix_cache" else None

    adapter._call_engine_async = call
    return adapter


def test_new_weights_invalidate_scheduler_and_encoder_caches():
    events = []
    adapter_with_cache(events).update_weights(7)
    assert events == ["publish", "reset_prefix_cache", "reset_encoder_cache"]


def test_failed_cache_reset_does_not_report_successful_update():
    with pytest.raises(RuntimeError, match="prefix cache reset failed"):
        adapter_with_cache([], reset_ok=False).update_weights(7)


def test_validation_keeps_failures_from_nonzero_dp_ranks():
    adapter = adapter_with_cache([])
    adapter._collective_rpc_all_dp_cores = lambda *args, **kwargs: [
        [{"expert0": True}],
        [{"expert1": False}],
    ]
    results = adapter.execute_task_in_model_worker("shardstream_execute")
    assert len(results) == 2
    assert not all(all(result.values()) for result in results)
