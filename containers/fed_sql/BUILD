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
    srcs = ["confidential_transform_server.cc"],
    hdrs = ["confidential_transform_server.h"],
    deps = [
        ":sensitive_columns",
        "//containers:blob_metadata",
        "//containers:confidential_transform_server_base",
        "//containers:crypto",
        "//containers:session",
        "//containers/sql:sqlite_adapter",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:fed_sql_container_config_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@gemma//:common",
        "@gemma//:gemma_lib",
        "@oak//proto/containers:orchestrator_crypto_cc_grpc",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_aggregator",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:config_converter",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_library(
    name = "sensitive_columns",
    srcs = ["sensitive_columns.cc"],
    hdrs = ["sensitive_columns.h"],
    deps = [
        "//containers:blob_metadata",
        "//containers:confidential_transform_server_base",
        "//containers:crypto",
        "//containers:session",
        "//containers/sql:sqlite_adapter",
        "@boringssl//:crypto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@federated-compute//fcp/base:status_converters",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
    ],
)

cc_test(
    name = "sensitive_columns_test",
    size = "small",
    srcs = ["sensitive_columns_test.cc"],
    deps = [
        ":sensitive_columns",
        "//containers:blob_metadata",
        "//containers:crypto",
        "//containers:crypto_test_utils",
        "//containers/sql:sqlite_adapter",
        "//testing:parse_text_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:fed_sql_container_config_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_aggregator",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_test(
    name = "confidential_transform_server_test",
    size = "small",
    srcs = ["confidential_transform_server_test.cc"],
    deps = [
        ":confidential_transform_server",
        "//containers:blob_metadata",
        "//containers:crypto",
        "//containers:crypto_test_utils",
        "//testing:parse_text_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:fed_sql_container_config_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_aggregator",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
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
