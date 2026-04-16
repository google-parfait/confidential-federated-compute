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

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "batched_inference_pi_oak_main",
    srcs = ["batched_inference_pi_oak_main.cc"],
    deps = [
        ":pi_batched_inference_provider",
        ":pi_client",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@confidential-federated-compute//containers/batched_inference:batched_inference_engine",
        "@confidential-federated-compute//containers/batched_inference:batched_inference_server",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

cc_library(
    name = "pi_batched_inference_provider",
    srcs = ["pi_batched_inference_provider.cc"],
    hdrs = ["pi_batched_inference_provider.h"],
    deps = [
        ":pi_client",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@confidential-federated-compute//containers/batched_inference:batched_inference_engine",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
    ],
)

cc_library(
    name = "oak_session_utils",
    srcs = ["oak_session_utils.cc"],
    hdrs = ["oak_session_utils.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@oak//cc/oak_session:client_session",
        "@oak//cc/oak_session:server_session",
        "@oak//proto/services:session_v1_service_cc_grpc",
    ],
)

cc_library(
    name = "pi_client",
    srcs = ["pi_client.cc"],
    hdrs = ["pi_client.h"],
    data = ["public_keys_prod_proto.binarypb"],
    deps = [
        ":oak_session_utils",
        ":session_config_rs",
        "//proto:private_aratea_service_cc_proto",
        "//proto:verification_keys_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@oak//cc/oak_session:client_session",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:server_session",
        "@oak//proto/services:session_v1_service_cc_grpc",
    ],
)

rust_library(
    name = "session_config_rs",
    srcs = ["session_config.rs"],
    deps = [
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_crypto",
        "@oak//oak_crypto:oak_crypto_tink",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak//oak_time",
        "@oak//oak_time:oak_time_std",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:ecdsa",
        "@oak_crates_index//:p256",
    ],
)

cc_test(
    name = "pi_client_test",
    srcs = ["pi_client_test.cc"],
    data = ["public_keys_prod_proto.binarypb"],
    deps = [
        ":pi_client",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
    ],
)

pkg_tar(
    name = "batched_inference_pi_oak_tar",
    srcs = [":batched_inference_pi_oak_main"],
    include_runfiles = True,
)

pkg_tar(
    name = "keys_tar",
    srcs = ["public_keys_prod_proto.binarypb"],
    mode = "0444",
    package_dir = "/private_inference",
)

oci_image(
    name = "batched_inference_pi_oak_oci_image",
    base = "@distroless_cc_debian12_base",
    cmd = [
        "--server_address=10.0.2.100",
    ],
    entrypoint = ["/batched_inference_pi_oak_main"],
    tars = [
        ":batched_inference_pi_oak_tar",
        ":keys_tar",
    ],
)

oci_runtime_bundle(
    name = "batched_inference_pi_oak_oci_runtime_bundle",
    image = ":batched_inference_pi_oak_oci_image",
)
