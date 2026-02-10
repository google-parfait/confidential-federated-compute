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

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
)

cc_library(
    name = "batched_inference_provider",
    srcs = [],
    hdrs = ["batched_inference_provider.h"],
    deps = ["@com_google_absl//absl/status:statusor"],
)

cc_library(
    name = "batched_inference_fn",
    srcs = ["batched_inference_fn.cc"],
    hdrs = ["batched_inference_fn.h"],
    deps = [
        ":batched_inference_provider",
        "//containers/fns",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:any_cc_proto",
    ],
)

cc_test(
    name = "batched_inference_fn_test",
    srcs = ["batched_inference_fn_test.cc"],
    deps = [
        ":batched_inference_fn",
        "//containers:session",
        "//containers/fns",
        "//containers/testing",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:any_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "batched_inference_server",
    srcs = [
        "batched_inference_server.cc",
    ],
    hdrs = [
        "batched_inference_server.h",
    ],
    deps = [
        ":batched_inference_fn",
        ":batched_inference_provider",
        "//containers/fns",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:any_cc_proto",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

cc_test(
    name = "batched_inference_server_test",
    srcs = ["batched_inference_server_test.cc"],
    defines = ["FCP_CLIENT_SUPPORT_CONFIDENTIAL_AGG"],
    deps = [
        ":batched_inference_provider",
        ":batched_inference_server",
        "//containers:crypto_test_utils",
        "//testing:parse_text_proto",
        "@boringssl//:crypto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googleapis//google/rpc:code_cc_proto",
        "@com_google_protobuf//:any_cc_proto",
        "@com_google_protobuf//:struct_cc_proto",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/confidentialcompute:constants",
        "@federated-compute//fcp/confidentialcompute:cose",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:kms_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@oak//cc/crypto:client_encryptor",
        "@oak//cc/crypto:common",
        "@oak//cc/crypto:encryption_key",
        "@oak//proto/crypto:crypto_cc_proto",
    ],
)

cc_binary(
    name = "main_for_testing",
    srcs = ["main_for_testing.cc"],
    deps = [
        ":batched_inference_provider",
        ":batched_inference_server",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

pkg_tar(
    name = "tar_for_testing",
    srcs = [":main_for_testing"],
    include_runfiles = True,
)

oci_image(
    name = "oci_image_for_testing",
    base = "@distroless_cc_debian12_base",
    cmd = [],
    entrypoint = ["/main_for_testing"],
    tars = [":tar_for_testing"],
)

oci_runtime_bundle(
    name = "oci_runtime_bundle_for_testing",
    image = ":oci_image_for_testing",
)
