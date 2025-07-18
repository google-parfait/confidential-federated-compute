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

load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_library", "rust_test")
load("@rules_rust_prost//:defs.bzl", "rust_prost_library")
load("//:proto_data.bzl", "proto_data")

package(features = ["layering_check"])

rust_library(
    name = "access_policies",
    srcs = ["access_policies.rs"],
    deps = [
        ":access_policy_prost_proto",
        ":kms_prost_proto",
        ":prost_proto_conversion",
        "@oak//oak_attestation_verification",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_proto_rust",
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
        ":session_test_utils",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:prost",
    ],
)

rust_prost_library(
    name = "access_policy_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:access_policy_proto",
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
    deps = [
        ":access_policies",
        ":key_derivation",
        ":kms_prost_proto",
        ":prost_proto_conversion",
        ":release_tokens",
        ":storage_client",
        ":storage_prost_proto",
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
        ":prost_proto_conversion",
        ":release_tokens",
        ":session_test_utils",
        ":storage",
        ":storage_client",
        ":storage_prost_proto",
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
)

proto_data(
    name = "insecure_reference_values",
    src = "insecure_reference_values.txtpb",
    out = "insecure_reference_values.binarypb",
    proto_deps = ["@oak//proto/attestation:reference_value_proto"],
    proto_name = "oak.attestation.v1.ReferenceValues",
)

rust_binary(
    name = "main",
    srcs = ["main.rs"],
    compile_data = [
        ":insecure_reference_values",
        "//reference_values/kms:reference_values",
    ],
    rustc_env = {
        "INSECURE_REFERENCE_VALUES": "$(location :insecure_reference_values)",
        "REFERENCE_VALUES": "$(location //reference_values/kms:reference_values)",
    },
    rustc_flags = ["-Cpanic=abort"],
    deps = [
        ":key_management_service",
        ":kms_prost_proto",
        ":session_v1_service_prost_proto",
        ":storage_actor",
        ":storage_client",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_proto_rust",
        "@oak//oak_sdk/common:oak_sdk_common",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:opentelemetry",
        "@oak_crates_index//:opentelemetry-appender-tracing",
        "@oak_crates_index//:opentelemetry-otlp",
        "@oak_crates_index//:opentelemetry_sdk",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:slog",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
        "@oak_crates_index//:tracing-slog",
        "@oak_crates_index//:tracing-subscriber",
        "@trusted_computations_platform//proto:tcp_proto",
        "@trusted_computations_platform//runtime:tcp_runtime",
    ],
)

rust_library(
    name = "matchers",
    testonly = True,
    srcs = ["matchers.rs"],
    deps = [
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:tonic",
    ],
)

rust_test(
    name = "matchers_test",
    crate = ":matchers",
)

oci_image(
    name = "oci_image",
    base = "@distroless_cc_debian12",
    entrypoint = ["/main"],
    tars = [":tar"],
)

oci_runtime_bundle(
    name = "oci_runtime_bundle",
    image = ":oci_image",
    visibility = ["//:__pkg__"],
)

rust_library(
    name = "prost_proto_conversion",
    srcs = ["prost_proto_conversion.rs"],
    deps = [
        ":access_policy_prost_proto",
        ":kms_prost_proto",
        ":session_v1_service_prost_proto",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:prost",
    ],
)

rust_library(
    name = "release_tokens",
    srcs = ["release_tokens.rs"],
    deps = [
        ":key_derivation",
        ":storage_prost_proto",
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
        ":storage_prost_proto",
        "@boringssl//rust/bssl-crypto",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:googletest",
    ],
)

rust_library(
    name = "session_config",
    srcs = ["session_config.rs"],
    deps = [
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_crypto",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
    ],
)

rust_library(
    name = "session_test_utils",
    testonly = True,
    srcs = ["session_test_utils.rs"],
    deps = [
        ":access_policy_prost_proto",
        ":kms_prost_proto",
        ":prost_proto_conversion",
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_sdk/common:oak_sdk_common",
        "@oak//oak_sdk/standalone:oak_sdk_standalone",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:p256",
        "@oak_crates_index//:rand_core",
    ],
)

rust_prost_library(
    name = "session_v1_service_prost_proto",
    proto = "@oak//proto/services:session_v1_service_proto",
)

rust_library(
    name = "storage",
    srcs = ["storage.rs"],
    deps = [
        ":storage_prost_proto",
        "@oak//oak_attestation_verification_types",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
    ],
)

rust_test(
    name = "storage_test",
    srcs = ["storage_test.rs"],
    env = {"RUST_LOG": "debug"},
    deps = [
        ":matchers",
        ":storage",
        ":storage_prost_proto",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:test-log",
        "@oak_crates_index//:tonic",
    ],
)

rust_library(
    name = "storage_actor",
    srcs = ["storage_actor.rs"],
    deps = [
        ":kms_prost_proto",
        ":prost_proto_conversion",
        ":session_config",
        ":storage",
        ":storage_prost_proto",
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:hashbrown",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:slog",
        "@oak_crates_index//:tonic",
        "@trusted_computations_platform//runtime:tcp_runtime",
    ],
)

rust_test(
    name = "storage_actor_test",
    srcs = ["storage_actor_test.rs"],
    env = {"RUST_LOG": "debug"},
    deps = [
        ":kms_prost_proto",
        ":prost_proto_conversion",
        ":session_config",
        ":session_test_utils",
        ":storage_actor",
        ":storage_prost_proto",
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:mockall",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:rand",
        "@oak_crates_index//:slog",
        "@oak_crates_index//:test-log",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing-slog",
        "@trusted_computations_platform//runtime:tcp_runtime",
    ],
)

rust_library(
    name = "storage_client",
    srcs = ["storage_client.rs"],
    rustc_flags = ["-Zpolonius"],
    deps = [
        ":prost_proto_conversion",
        ":session_config",
        ":session_v1_service_prost_proto",
        ":storage_prost_proto",
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:hashbrown",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tokio-util",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
    ],
)

rust_test(
    name = "storage_client_test",
    srcs = ["storage_client_test.rs"],
    env = {"RUST_LOG": "debug"},
    deps = [
        ":matchers",
        ":prost_proto_conversion",
        ":session_config",
        ":session_test_utils",
        ":session_v1_service_prost_proto",
        ":storage_client",
        ":storage_prost_proto",
        "@oak//oak_attestation_types",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:mockall",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:test-log",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tokio-util",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
    ],
)

# buildifier: disable=native-proto
proto_library(
    name = "storage_proto",
    srcs = ["storage.proto"],
    data = [":storage_transform"],
    deps = [
        "@com_google_googleapis//google/rpc:status_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
    ],
)

rust_prost_library(
    name = "storage_prost_proto",
    proto = ":storage_proto",
)

pkg_tar(
    name = "tar",
    srcs = [":main"],
)
