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

import pytest

from awex.transfer.nccl_device import (
    NCCLDeviceUnavailableError,
    _sequence_from_step,
)


def test_sequence_maps_initial_and_training_steps_to_positive_values():
    assert _sequence_from_step(-1) == 1
    assert _sequence_from_step(0) == 2
    assert _sequence_from_step(7) == 9


def test_sequence_rejects_steps_before_initial_weight_load():
    with pytest.raises(NCCLDeviceUnavailableError, match="at least -1"):
        _sequence_from_step(-2)
