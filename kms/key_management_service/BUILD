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
    name = "access_policies",
    srcs = ["access_policies.rs"],
    deps = [
        ":access_policy_prost_proto",
        ":kms_prost_proto",
        "//:prost_proto_conversion",
        "@oak//oak_attestation_verification",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_proto_rust",
        "@oak//oak_time",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:prost",
    ],
)

rust_test(
    name = "access_policies_test",
    srcs = ["access_policies_test.rs"],
    deps = [
        ":access_policies",
        ":access_policy_prost_proto",
        "//testing:session_test_utils",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:prost",
    ],
)

rust_prost_library(
    name = "access_policy_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:access_policy_proto",
    visibility = ["//:__subpackages__"],
)

rust_library(
    name = "key_derivation",
    srcs = ["key_derivation.rs"],
    deps = [
        "@boringssl//rust/bssl-crypto",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:futures",
    ],
)

rust_test(
    name = "key_derivation_test",
    srcs = ["key_derivation_test.rs"],
    proc_macro_deps = ["@oak_crates_index//:async-trait"],
    deps = [
        ":key_derivation",
        "@boringssl//rust/bssl-crypto",
        "@oak//oak_proto_rust",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:tokio",
    ],
)

rust_library(
    name = "key_management_service",
    srcs = ["key_management_service.rs"],
    visibility = ["//:__pkg__"],
    deps = [
        ":access_policies",
        ":key_derivation",
        ":kms_prost_proto",
        ":release_tokens",
        "//:prost_proto_conversion",
        "//storage_service:storage_client",
        "//storage_service:storage_prost_proto",
        "@boringssl//rust/bssl-crypto",
        "@oak//oak_crypto",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:hashbrown",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:rand",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
    ],
)

rust_test(
    name = "key_management_service_test",
    srcs = ["key_management_service_test.rs"],
    proc_macro_deps = ["@oak_crates_index//:async-trait"],
    deps = [
        ":access_policy_prost_proto",
        ":key_derivation",
        ":key_management_service",
        ":kms_prost_proto",
        ":release_tokens",
        "//:prost_proto_conversion",
        "//storage_service:storage",
        "//storage_service:storage_client",
        "//storage_service:storage_prost_proto",
        "//testing:session_test_utils",
        "@boringssl//rust/bssl-crypto",
        "@oak//oak_crypto",
        "@oak//oak_proto_rust",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tonic",
    ],
)

rust_prost_library(
    name = "kms_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:kms_proto",
    visibility = ["//:__subpackages__"],
)

rust_library(
    name = "release_tokens",
    srcs = ["release_tokens.rs"],
    deps = [
        ":key_derivation",
        "//storage_service:storage_prost_proto",
        "@boringssl//rust/bssl-crypto",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:hashbrown",
    ],
)

rust_test(
    name = "release_tokens_test",
    srcs = ["release_tokens_test.rs"],
    deps = [
        ":key_derivation",
        ":release_tokens",
        "//storage_service:storage_prost_proto",
        "@boringssl//rust/bssl-crypto",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:googletest",
    ],
)
