# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//:proto_data.bzl", "proto_data")

package(default_visibility = ["//ledger_service:__pkg__"])

proto_data(
    name = "endorsements",
    testonly = True,
    src = "endorsements.txtpb",
    out = "endorsements.binarypb",
    proto_deps = ["@oak//proto/attestation:endorsement_proto"],
    proto_name = "oak.attestation.v1.Endorsements",
)

proto_data(
    name = "evidence",
    testonly = True,
    src = "evidence.txtpb",
    out = "evidence.binarypb",
    proto_deps = ["@oak//proto/attestation:evidence_proto"],
    proto_name = "oak.attestation.v1.Evidence",
)
