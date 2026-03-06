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
        ":payload_signer",
        "//:prost_proto_conversion",
        "@boringssl//rust/bssl-crypto",
        "@oak//oak_attestation_verification",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_proto_rust",
        "@oak//oak_sdk/common:oak_sdk_common",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak//oak_session",
        "@oak//oak_session/endorsed_evidence:oak_session_endorsed_evidence",
        "@oak//oak_time",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:ciborium",
        "@oak_crates_index//:integer-encoding",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:rand",
        "@oak_crates_index//:serde",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tonic",
    ],
)

rust_test(
    name = "attestation_transparency_service_test",
    srcs = ["attestation_transparency_service_test.rs"],
    proc_macro_deps = ["@oak_crates_index//:async-trait"],
    deps = [
        ":attestation_transparency_service",
        ":attestation_transparency_service_prost_proto",
        ":payload_signer",
        "//testing:matchers",
        "@boringssl//rust/bssl-crypto",
        "@oak//oak_crypto",
        "@oak//oak_proto_rust",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak//oak_sdk/standalone:oak_sdk_standalone",
        "@oak//oak_session",
        "@oak//oak_time",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:p256",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:rand_core",
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

rust_library(
    name = "payload_signer",
    srcs = ["payload_signer.rs"],
    visibility = ["//key_management_service:__subpackages__"],
    deps = [
        ":attestation_transparency_service_prost_proto",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:mockall",
    ],
)
