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

package(default_visibility = ["//:__subpackages__"])

load("@com_github_grpc_grpc//bazel:grpc_build_system.bzl", "grpc_cc_test")
load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":pipeline_transform_server",
        "//containers:oak_orchestrator_client",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@oak//oak_containers/proto:interfaces_cc_grpc",
        "@oak//oak_containers/proto:interfaces_cc_proto",
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
    name = "sql_server_oci_filesystem_bundle",
    image = ":oci_image",
)

cc_library(
    name = "sqlite_adapter",
    srcs = ["sqlite_adapter.cc"],
    hdrs = ["sqlite_adapter.h"],
    deps = [
        ":sql_data_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_parser",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/client:example_query_result_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@sqlite_bazel//:sqlite3",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@federated-compute//fcp/protos:plan_cc_proto"
    ],
)

cc_test(
    name = "sqlite_adapter_test",
    size = "small",
    srcs = ["sqlite_adapter_test.cc"],
    deps = [
        ":sqlite_adapter",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_builder",
        "@federated-compute//fcp/aggregation/testing:test_data",
        "@googletest//:gtest_main",
        "@sqlite_bazel//:sqlite3",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
    ],
)

cc_proto_library(
    name = "sql_data_cc_proto",
    deps = [":sql_data_proto"],
)

proto_library(
    name = "sql_data_proto",
    srcs = ["sql_data.proto"],
    deps = ["@federated-compute//fcp/client:example_query_result_proto"],
)

cc_library(
    name = "sql_data_converter",
    srcs = ["sql_data_converter.cc"],
    hdrs = ["sql_data_converter.h"],
    deps = [
        ":sql_data_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@federated-compute//fcp/aggregation/core:tensor",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_builder",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_parser",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/client:example_query_result_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
    ],
)

cc_test(
    name = "sql_data_converter_test",
    size = "small",
    srcs = ["sql_data_converter_test.cc"],
    deps = [
        ":sql_data_converter",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_builder",
        "@federated-compute//fcp/aggregation/testing:test_data",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "pipeline_transform_server",
    srcs = ["pipeline_transform_server.cc"],
    hdrs = ["pipeline_transform_server.h"],
    deps = [
        "@com_google_protobuf//:protobuf",
        ":sql_data_cc_proto",
        ":sql_data_converter",
        ":sqlite_adapter",
        "//containers:crypto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
    ],
)

cc_test(
    name = "pipeline_transform_server_test",
    size = "small",
    srcs = ["pipeline_transform_server_test.cc"],
    deps = [
        ":pipeline_transform_server",
        ":sql_data_cc_proto",
        "//containers:crypto",
        "//containers:crypto_test_utils",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_builder",
        "@federated-compute//fcp/aggregation/testing:test_data",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@googletest//:gtest_main",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
    ],
)

cc_test(
    name = "pipeline_transform_server_benchmarks",
    size = "small",
    srcs = ["pipeline_transform_server_benchmarks.cc"],
    tags = ["manual"],
    deps = [
        ":pipeline_transform_server",
        ":sql_data_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_builder",
        "@federated-compute//fcp/aggregation/testing:test_data",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@google_benchmark//:benchmark",
        "@googletest//:gtest_main",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
    ],
)
