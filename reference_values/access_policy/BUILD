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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//:proto_data.bzl", "proto_data")

package(features = ["layering_check"])

build_test(
    name = "access_policy_build_test",
    targets = [
        ":gboard",
    ],
)

proto_data(
    name = "gboard",
    src = "gboard.txtpb",
    proto_deps = [
        "@federated-compute//fcp/protos/confidentialcompute:access_policy_endorsement_options_proto",
    ],
    proto_name = "fcp.confidentialcompute.AccessPolicyEndorsementOptions",
)
