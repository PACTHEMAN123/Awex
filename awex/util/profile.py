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

import json
import os
from typing import Any


def profile_enabled() -> bool:
    return os.environ.get("AWEX_PROFILE", "0").lower() in {"1", "true", "yes"}


def profile_phase(step_id: int) -> str:
    warmup_updates = int(os.environ.get("AWEX_PROFILE_WARMUP_UPDATES", "0"))
    update_index = int(step_id) + 1
    return "warmup" if update_index < warmup_updates else "measure"


def emit_profile(logger, **values: Any) -> None:
    if not profile_enabled():
        return
    record = (
        "AWEX_PROFILE "
        + json.dumps(values, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    try:
        # One write keeps records intact when torchrun ranks share stdout.
        os.write(1, record)
    except OSError:
        logger.info("%s", record.decode().rstrip())
