# Copyright 2024 Google LLC.
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

cc_library(
    name = "confidential_transform_server",
    hdrs = ["confidential_transform_server.h"],
    deps = [
        "//containers:blob_metadata",
        "//containers:confidential_transform_server_base",
        "//containers:crypto",
        "//containers:session",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@oak//proto/containers:orchestrator_crypto_cc_grpc",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/base",
    ],
)

cc_test(
    name = "confidential_transform_server_test",
    srcs = ["confidential_transform_server_test.cc"],
    deps = [
        "confidential_transform_server",
        "//containers:blob_metadata",
        "//containers:crypto",
        "//containers:crypto_test_utils",
        "//containers:session",
        "//testing:parse_text_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@googletest//:gtest_main",
        "@oak//proto/containers:orchestrator_crypto_cc_grpc",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/base",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":confidential_transform_server",
        "//containers:oak_orchestrator_client",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//proto/containers:interfaces_cc_grpc",
        "@oak//proto/containers:orchestrator_crypto_cc_grpc",
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
