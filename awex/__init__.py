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

from importlib import import_module

__all__ = [
    "InferenceConfig",
    "NCCLWeightsWriter",
    "WeightsReader",
    "NCCLWorkerWeightsReader",
]

_LAZY_EXPORTS = {
    "InferenceConfig": ("awex.config", "InferenceConfig"),
    "NCCLWeightsWriter": ("awex.writer.nccl_writer", "NCCLWeightsWriter"),
    "WeightsReader": ("awex.reader.weights_reader", "WeightsReader"),
    "NCCLWorkerWeightsReader": (
        "awex.reader.nccl_reader",
        "NCCLWorkerWeightsReader",
    ),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
