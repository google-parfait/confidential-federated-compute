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

load("@rules_rust//proto/prost:defs.bzl", "rust_prost_library")
load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

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

rust_library(
    name = "session_config",
    srcs = ["session_config.rs"],
    deps = [
        "@oak//oak_attestation_types",
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
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_proto_rust",
        "@oak//oak_restricted_kernel_sdk",
        "@oak_crates_index//:anyhow",
    ],
)

rust_library(
    name = "storage",
    srcs = ["storage.rs"],
    deps = [
        ":storage_prost_proto",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:log",
        "@oak_crates_index//:tonic",
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
        ":session_config",
        ":storage",
        ":storage_prost_proto",
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_crypto",
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
        ":session_config",
        ":session_test_utils",
        ":storage_actor",
        ":storage_prost_proto",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:mockall",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:rand",
        "@oak_crates_index//:slog",
        "@oak_crates_index//:slog-stdlog",
        "@oak_crates_index//:test-log",
        "@oak_crates_index//:tonic",
        "@trusted_computations_platform//runtime:tcp_runtime",
    ],
)

rust_library(
    name = "storage_client",
    srcs = ["storage_client.rs"],
    rustc_flags = ["-Zpolonius"],
    deps = [
        ":session_config",
        ":session_v1_service_prost_proto",
        ":storage_prost_proto",
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_crypto",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:hashbrown",
        "@oak_crates_index//:log",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tokio-util",
        "@oak_crates_index//:tonic",
    ],
)

rust_test(
    name = "storage_client_test",
    srcs = ["storage_client_test.rs"],
    env = {"RUST_LOG": "debug"},
    deps = [
        ":matchers",
        ":session_config",
        ":session_test_utils",
        ":session_v1_service_prost_proto",
        ":storage_client",
        ":storage_prost_proto",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:log",
        "@oak_crates_index//:mockall",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:test-log",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tokio-util",
        "@oak_crates_index//:tonic",
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
        "@oak//proto/session:session_proto",
    ],
)

rust_prost_library(
    name = "storage_prost_proto",
    proto = ":storage_proto",
)

rust_prost_library(
    name = "session_v1_service_prost_proto",
    proto = "@oak//proto/services:session_v1_service_proto",
)
