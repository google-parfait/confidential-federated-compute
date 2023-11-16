# Copyright 2023 The Confidential Federated Compute Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(default_visibility = ["//:__subpackages__"])

load("@rules_cc//cc:defs.bzl", "cc_proto_library")

cc_library(
    name = "sqlite_adapter",
    srcs = ["sqlite_adapter.cc"],
    hdrs = ["sqlite_adapter.h"],
    deps = [
        ":sql_data_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_parser",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/client:example_query_result_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@sqlite_bazel//:sqlite3",
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
        "@com_google_absl//absl/log",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_parser",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/client:example_query_result_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
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
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@federated-compute//fcp/aggregation/testing:test_data",
        "@googletest//:gtest_main",
    ],
)
