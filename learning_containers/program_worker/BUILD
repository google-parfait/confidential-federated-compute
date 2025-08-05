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
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_rust//rust:defs.bzl", "rust_library")
load("@rules_rust_prost//:defs.bzl", "rust_prost_library")

package(features = ["layering_check"])

cc_library(
    name = "program_worker_server",
    srcs = ["program_worker_server.cc"],
    hdrs = ["program_worker_server.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/confidentialcompute:tff_execution_helper",
        "@federated-compute//fcp/protos/confidentialcompute:program_worker_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:program_worker_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_config_cc_proto",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:server_session",
        "@oak//proto/session:session_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_test(
    name = "program_worker_server_test",
    size = "small",
    srcs = ["program_worker_server_test.cc"],
    data = [
        "//program_worker/testing:test_tff_values",
    ],
    deps = [
        ":program_worker_server",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/protos/confidentialcompute:program_worker_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:program_worker_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_config_cc_proto",
        "@federated-compute//fcp/testing:parse_text_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@oak//cc/ffi:bytes_bindings",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/ffi:error_bindings",
        "@oak//cc/oak_session:client_session",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
    ],
)

rust_prost_library(
    name = "session_prost_proto",
    proto = "@oak//proto/session:messages_proto",
)

rust_library(
    name = "session_config",
    srcs = ["session_config.rs"],
    deps = [
        ":session_prost_proto",
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification/dice_attestation_verification:oak_dice_attestation_verifier",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_crypto",
        "@oak//oak_proto_rust",
        "@oak//oak_sdk/common:oak_sdk_common",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak//oak_session",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:tokio",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    tags = ["noasan"],
    deps = [
        ":program_worker_server",
        ":session_config",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/ffi:bytes_bindings",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/ffi:error_bindings",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
        "@oak//proto/session:session_cc_proto",
    ],
)

pkg_tar(
    name = "tar",
    srcs = [":main"],
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
