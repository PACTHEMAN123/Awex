import argparse

import pytest

from awex.tests.weights_exchange_multi_vllm_it import (
    MultiVLLMWeightsExchangeIT,
    _inference_endpoint,
)


def test_inference_endpoint_parses_rank_host_and_port():
    assert _inference_endpoint("3,10.0.0.2,18001") == (3, "10.0.0.2", 18001)


@pytest.mark.parametrize(
    "value",
    (
        "missing-fields",
        "-1,10.0.0.2,18001",
        "0,,18001",
        "0,10.0.0.2,0",
        "0,10.0.0.2,65536",
    ),
)
def test_inference_endpoint_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _inference_endpoint(value)


def test_publication_endpoints_use_multi_host_override():
    harness = object.__new__(MultiVLLMWeightsExchangeIT)
    harness.inference_endpoints = [
        (0, "10.0.0.2", 18000),
        (1, "10.0.0.3", 18000),
    ]
    harness.inference_config = {"num_engines": 2}
    harness.host = "127.0.0.1"
    harness.port = 8000

    assert harness.publication_endpoints() == harness.inference_endpoints
