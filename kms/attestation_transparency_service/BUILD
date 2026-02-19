# Copyright 2026 Google LLC.
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

load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")
load("@rules_rust_prost//:defs.bzl", "rust_prost_library")

package(features = ["layering_check"])

rust_library(
    name = "attestation_transparency_service",
    srcs = ["attestation_transparency_service.rs"],
    visibility = ["//:__pkg__"],
    deps = [
        ":attestation_transparency_service_prost_proto",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tonic",
    ],
)

rust_test(
    name = "attestation_transparency_service_test",
    srcs = ["attestation_transparency_service_test.rs"],
    deps = [
        ":attestation_transparency_service",
        ":attestation_transparency_service_prost_proto",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tokio-util",
        "@oak_crates_index//:tonic",
    ],
)

rust_prost_library(
    name = "attestation_transparency_service_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:attestation_transparency_service_proto",
    visibility = ["//:__pkg__"],
)
