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
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

package(default_visibility = ["//:__subpackages__"])

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":pipeline_transform_server",
        "//containers:oak_orchestrator_client",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
)

cc_library(
    name = "sqlite_adapter",
    srcs = ["sqlite_adapter.cc"],
    hdrs = ["sqlite_adapter.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/client:example_query_result_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
        "@sqlite",
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
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "@sqlite",
    ],
)

cc_library(
    name = "pipeline_transform_server",
    srcs = ["pipeline_transform_server.cc"],
    hdrs = ["pipeline_transform_server.h"],
    deps = [
        ":sqlite_adapter",
        "//containers:crypto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@oak//proto/containers:orchestrator_crypto_cc_grpc",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_test(
    name = "pipeline_transform_server_test",
    size = "small",
    srcs = ["pipeline_transform_server_test.cc"],
    deps = [
        ":pipeline_transform_server",
        "//containers:crypto",
        "//containers:crypto_test_utils",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_test(
    name = "pipeline_transform_server_benchmarks",
    size = "small",
    srcs = ["pipeline_transform_server_benchmarks.cc"],
    tags = ["manual"],
    deps = [
        ":pipeline_transform_server",
        "//containers:crypto",
        "//containers:crypto_test_utils",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@google_benchmark//:benchmark",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)
