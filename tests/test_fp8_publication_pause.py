"""vLLM scheduler acknowledgements are required before publication begins."""
import pytest
from shardstream_benchmarks.fp8_config import set_publication_pause


@pytest.mark.parametrize("paused,status,path", [
    (True, "paused", "pause?mode=wait"),
    (False, "resumed", "resume"),
])
def test_pause_lifecycle_reaches_every_instance(paused, status, path):
    calls = []

    def request(url, payload):
        calls.append((url, payload))
        return {"status": status}

    endpoints = ["http://inference-a:8000", "http://inference-b:8000"]
    result = set_publication_pause(endpoints, paused=paused, request=request)
    assert calls == [(f"{url}/{path}", {}) for url in endpoints]
    assert result == [{"status": status}] * 2


def test_pause_rejects_missing_acknowledgement():
    with pytest.raises(RuntimeError, match="did not acknowledge paused"):
        set_publication_pause(["http://inference:8000"], paused=True,
                              request=lambda *args: {"error": "scheduler busy"})


def test_pause_rejects_empty_instance_set():
    with pytest.raises(ValueError, match="requires inference endpoints"):
        set_publication_pause([], paused=True, request=lambda *args: {})


def test_pause_propagates_connection_failure_before_publication():
    def unavailable(*args):
        raise ConnectionError("inference unavailable")

    with pytest.raises(ConnectionError):
        set_publication_pause(["http://inference:8000"], paused=True,
                              request=unavailable)
