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


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class InferenceConfigValidationError(ValueError):
    """Raised when InferenceConfig contains invalid or inconsistent settings."""

    pass


_VALID_COMM_BACKENDS: frozenset = frozenset({"transport"})


@dataclass
class InferenceConfig:
    """
    Configuration for inference.
    """

    model_path: Optional[str] = None
    tp_size: Optional[int] = None
    pp_size: Optional[int] = None
    dp_size: Optional[int] = None
    load_balance_method: Optional[str] = None
    ep_size: Optional[int] = None
    enable_dp_attention: Optional[bool] = None
    enable_dp_lm_head: Optional[bool] = None
    moe_dense_tp_size: Optional[int] = None
    nnodes: Optional[int] = None
    node_rank: Optional[int] = None
    local_rank: Optional[int] = None
    num_engines: int = 1
    engine_rank: int = 0
    meta_server_addr: Optional[str] = None
    comm_backend: str = "transport"
    weights_validation_steps: int = 0
    validate_weights_every_n_steps: int = 1
    dump_weights_list_for_validation: List[str] = field(default_factory=list)
    dump_weights_dir_for_validation: Optional[str] = None
    disable_weights_exchange_pipeline: bool = False
    enable_debug_mode: bool = False
    debug_mode_config: Dict = field(
        default_factory=dict, metadata={"help": "Debug mode configuration"}
    )
    weights_comm_nccl_group_size: int = 1

    def validate(self) -> None:
        """Validate configuration fields for consistency and correctness.

        Raises:
            InferenceConfigValidationError: if any field is invalid or fields are inconsistent.
        """
        errors: List[str] = []
        if self.comm_backend not in _VALID_COMM_BACKENDS:
            errors.append(
                f"comm_backend must be one of {sorted(_VALID_COMM_BACKENDS)}, got {self.comm_backend!r}"
            )
        if self.num_engines < 1:
            errors.append(f"num_engines must be >= 1, got {self.num_engines}")
        elif not 0 <= self.engine_rank < self.num_engines:
            errors.append(
                f"engine_rank must be in [0, num_engines), got engine_rank={self.engine_rank} with num_engines={self.num_engines}"
            )
        for name, val in [
            ("tp_size", self.tp_size),
            ("pp_size", self.pp_size),
            ("dp_size", self.dp_size),
            ("ep_size", self.ep_size),
            ("moe_dense_tp_size", self.moe_dense_tp_size),
            ("weights_comm_nccl_group_size", self.weights_comm_nccl_group_size),
        ]:
            if val is not None and val < 1:
                errors.append(f"{name} must be >= 1 when set, got {val}")
        if self.nnodes is not None:
            if self.nnodes < 1:
                errors.append(f"nnodes must be >= 1 when set, got {self.nnodes}")
            elif self.node_rank is not None and (not 0 <= self.node_rank < self.nnodes):
                errors.append(
                    f"node_rank must be in [0, nnodes), got node_rank={self.node_rank} with nnodes={self.nnodes}"
                )
        if self.validate_weights_every_n_steps < 1:
            errors.append(
                f"validate_weights_every_n_steps must be >= 1, got {self.validate_weights_every_n_steps}"
            )
        if self.weights_validation_steps < 0:
            errors.append(
                f"weights_validation_steps must be >= 0, got {self.weights_validation_steps}"
            )
        if self.dump_weights_list_for_validation and (
            not self.dump_weights_dir_for_validation
        ):
            errors.append(
                "dump_weights_dir_for_validation must be set when dump_weights_list_for_validation is non-empty"
            )
        if self.num_engines > 1 and (not self.meta_server_addr):
            errors.append(
                f"meta_server_addr must be set when num_engines > 1 and comm_backend is {self.comm_backend!r}"
            )
        if errors:
            msg = "InferenceConfig validation failed:\n" + "\n".join(
                (f"  - {e}" for e in errors)
            )
            raise InferenceConfigValidationError(msg)

    def validated(self) -> "InferenceConfig":
        """Call validate() and return self for easy chaining.

        Example::

            config = InferenceConfig.from_dict(d).validated()
        """
        self.validate()
        return self

    @staticmethod
    def from_dict(
        config_dict: Dict[str, Any], validate: bool = True
    ) -> "InferenceConfig":
        config_dict = {
            k: v
            for (k, v) in config_dict.items()
            if k in InferenceConfig.__dataclass_fields__
        }
        cfg = InferenceConfig(**config_dict)
        if validate:
            cfg.validate()
        return cfg
