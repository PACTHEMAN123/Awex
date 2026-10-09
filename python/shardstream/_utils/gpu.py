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


import subprocess

import torch

from shardstream import logging
from shardstream._utils import device as device_util
from shardstream._utils.common import pretty_bytes

logger = logging.getLogger(__name__)


def get_gpu_status() -> str:
    """Get accelerator status information in CSV format."""
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv",
            ],
            text=True,
        )
    except subprocess.CalledProcessError as e:
        return f"Failed to get GPU status via nvidia-smi: {e}"
    except FileNotFoundError:
        return "nvidia-smi not found; GPU status unavailable."


def print_current_gpu_status(stage):
    device_type = device_util.get_device_type()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    (mem_free, mem_total) = torch.cuda.mem_get_info()
    occupy = mem_total - mem_free
    logger.info(
        f"Device {device_type} memory status for [{stage}]: torch allocated {pretty_bytes(allocated)}, torch reserved {pretty_bytes(reserved)} device mem_free {pretty_bytes(mem_free)}, device occupy {pretty_bytes(occupy)}"
    )
