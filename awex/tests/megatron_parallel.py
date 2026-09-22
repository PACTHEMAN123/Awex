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

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MegatronParallelism:
    tp_size: int
    pp_size: int
    ep_size: int
    expert_tp_size: int

    @property
    def dense_parallel_size(self) -> int:
        return self.tp_size * self.pp_size

    @property
    def expert_parallel_size(self) -> int:
        return self.expert_tp_size * self.ep_size * self.pp_size

    @property
    def required_world_size_multiple(self) -> int:
        return math.lcm(self.dense_parallel_size, self.expert_parallel_size)

    def validate_world_size(self, world_size: int) -> None:
        if (
            world_size % self.dense_parallel_size != 0
            or world_size % self.expert_parallel_size != 0
        ):
            raise RuntimeError(
                "Invalid Megatron parallel config for WORLD_SIZE. "
                f"dense(tp*pp)={self.dense_parallel_size}, "
                "expert(expert_tp*ep*pp)="
                f"{self.expert_parallel_size}, WORLD_SIZE={world_size}, "
                f"required multiple={self.required_world_size_multiple}."
            )


def resolve_megatron_parallelism(
    *,
    tp_size: int,
    ep_size: int = 1,
    expert_tp_size: Optional[int] = None,
    pp_size: int = 1,
) -> MegatronParallelism:
    resolved_expert_tp_size = tp_size if expert_tp_size is None else expert_tp_size
    values = {
        "tp_size": tp_size,
        "pp_size": pp_size,
        "ep_size": ep_size,
        "expert_tp_size": resolved_expert_tp_size,
    }
    for name, value in values.items():
        if value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    return MegatronParallelism(**values)
