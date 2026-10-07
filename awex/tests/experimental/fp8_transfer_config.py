# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
