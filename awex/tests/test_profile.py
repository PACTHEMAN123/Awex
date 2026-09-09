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

from awex.util.profile import profile_phase


def test_profile_phase_marks_only_configured_initial_updates_as_warmup(
    monkeypatch,
):
    monkeypatch.setenv("AWEX_PROFILE_WARMUP_UPDATES", "2")

    assert profile_phase(-1) == "warmup"
    assert profile_phase(0) == "warmup"
    assert profile_phase(1) == "measure"


def test_profile_phase_defaults_to_measure(monkeypatch):
    monkeypatch.delenv("AWEX_PROFILE_WARMUP_UPDATES", raising=False)

    assert profile_phase(-1) == "measure"
