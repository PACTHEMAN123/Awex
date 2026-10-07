"""Native vLLM FP8 storage for the fused device-v2 transfer experiment."""

import json
import os


def fp8_server_args() -> list[str]:
    if os.environ.get("AWEX_NCCL_DEVICE_V2_FP8_BLOCKWISE", "0") != "1":
        return []
    block = [
        int(os.environ.get("AWEX_NCCL_DEVICE_V2_FP8_BLOCK_ROWS", "128")),
        int(os.environ.get("AWEX_NCCL_DEVICE_V2_FP8_BLOCK_COLS", "128")),
    ]
    # Dummy initializes the target's native FP8 parameter/scale layout. BF16
    # checkpoint weights arrive from the training model through the fused FIFO
    # path before generating; the shared checkpoint config is never edited.
    overrides = {
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": block,
            "ignored_layers": ["lm_head", "model.layers.*.mlp.gate"],
        }
    }
    return [
        "--quantization",
        "fp8",
        "--load-format",
        "dummy",
        "--hf-overrides",
        json.dumps(overrides),
        "--kernel-config",
        json.dumps({"moe_backend": "triton", "linear_backend": "triton"}),
    ]
