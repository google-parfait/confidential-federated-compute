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

load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(features = ["layering_check"])

proto_library(
    name = "willow_messages_proto",
    srcs = ["willow_messages.proto"],
)

# buildifier: disable=native-cc-proto
cc_proto_library(
    name = "willow_messages_cc_proto",
    deps = [":willow_messages_proto"],
)

cc_library(
    name = "service",
    srcs = [
        "willow_session.cc",
        "willow_transform_service.cc",
    ],
    hdrs = [
        "willow_session.h",
        "willow_transform_service.h",
    ],
    deps = [
        ":willow_messages_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:any_cc_proto",
        "@confidential-federated-compute//containers:blob_metadata",
        "@confidential-federated-compute//containers:session",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@secure_aggregation//willow/api:server_accumulator",
        "@secure_aggregation//willow/api:server_accumulator_cxx",  # fixdeps: keep
        "@secure_aggregation//willow/proto/willow:aggregation_config_cc_proto",
        "@secure_aggregation//willow/proto/willow:messages_cc_proto",
        "@secure_aggregation//willow/proto/willow:server_accumulator_cc_proto",
    ],
)

cc_test(
    name = "willow_transform_service_test",
    srcs = ["willow_transform_service_test.cc"],
    deps = [
        ":service",
        ":willow_messages_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@confidential-federated-compute//containers:crypto_test_utils",
        "@federated-compute//fcp/base:random_token",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@secure_aggregation//willow/api:client_cc",
        "@secure_aggregation//willow/api:server_accumulator",
        "@secure_aggregation//willow/api:server_accumulator_cxx",  # fixdeps: keep
        "@secure_aggregation//willow/input_encoding:codec",
        "@secure_aggregation//willow/proto/willow:aggregation_config_cc_proto",
        "@secure_aggregation//willow/testing_utils:shell_testing_decryptor_cc",
        "@secure_aggregation//willow/testing_utils:testing_utils_cc",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":service",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@oak//cc/containers/sdk:orchestrator_client",
    ],
)

pkg_tar(
    name = "tar",
    srcs = [":main"],
)

oci_image(
    name = "oci_image",
    base = "@distroless_cc_debian12_base",
    entrypoint = ["/main"],
    tars = [":tar"],
)

oci_runtime_bundle(
    name = "oci_runtime_bundle",
    image = ":oci_image",
    visibility = ["//:__pkg__"],
)
