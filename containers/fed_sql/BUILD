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
load("@rules_proto//proto:defs.bzl", "proto_library")

cc_library(
    name = "confidential_transform_server",
    srcs = ["confidential_transform_server.cc"],
    hdrs = ["confidential_transform_server.h"],
    deps = [
        ":inference_model",
        ":kms_session",
        ":ledger_session",
        ":private_state",
        ":sensitive_columns",
        ":session_utils",
        "//containers:confidential_transform_server_base",
        "//containers:crypto",
        "//containers:session",
        "//containers/sql:sqlite_adapter",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/confidentialcompute:private_state",
        "@federated-compute//fcp/protos/confidentialcompute:access_policy_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:fed_sql_container_config_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@oak//cc/crypto:signing_key",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_aggregator",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:config_converter",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_library(
    name = "inference_model",
    srcs = ["inference_model.cc"],
    hdrs = ["inference_model.h"],
    deps = [
        "//containers/sql:sqlite_adapter",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@gemma//:common",
        "@gemma//:gemma_lib",
        "@gemma//:threading",
        "@gemma//compression:io",
        "@gemma//compression:sfp",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
    ],
)

cc_test(
    name = "inference_model_test",
    size = "small",
    srcs = ["inference_model_test.cc"],
    deps = [
        ":inference_model",
        "//containers/sql:sqlite_adapter",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@gemma//:gemma_lib",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
    ],
)

cc_library(
    name = "sensitive_columns",
    srcs = ["sensitive_columns.cc"],
    hdrs = ["sensitive_columns.h"],
    deps = [
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
        ":budget_cc_proto",
        ":confidential_transform_server",
        ":inference_model",
        "//containers:crypto",
        "//containers:crypto_test_utils",
        "//containers/fed_sql/testing:mocks",
        "//containers/fed_sql/testing:test_utils",
        "//testing:matchers",
        "//testing:parse_text_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/confidentialcompute:private_state",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:fed_sql_container_config_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:kms_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@googletest//:gtest_main",
        "@oak//cc/crypto:client_encryptor",
        "@oak//cc/crypto:encryption_key",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_aggregator",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_test(
    name = "session_utils_test",
    srcs = ["session_utils_test.cc"],
    deps = [
        ":inference_model",
        ":session_utils",
        "//containers/fed_sql/testing:test_utils",
        "//testing:matchers",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@gemma//:gemma_lib",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_library(
    name = "session_utils",
    srcs = ["session_utils.cc"],
    hdrs = ["session_utils.h"],
    deps = [
        ":inference_model",
        "//containers:crypto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@gemma//:gemma_lib",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
    ],
)

cc_library(
    name = "ledger_session",
    srcs = ["ledger_session.cc"],
    hdrs = ["ledger_session.h"],
    deps = [
        ":inference_model",
        ":sensitive_columns",
        ":session_utils",
        "//containers:session",
        "//containers/sql:sqlite_adapter",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:fed_sql_container_config_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_aggregator",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_library(
    name = "kms_session",
    srcs = ["kms_session.cc"],
    hdrs = ["kms_session.h"],
    deps = [
        ":inference_model",
        ":private_state",
        ":range_tracker",
        ":sensitive_columns",
        ":session_utils",
        "//containers:big_endian",
        "//containers:crypto",
        "//containers:session",
        "//containers/sql:sqlite_adapter",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/confidentialcompute:cose",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:fed_sql_container_config_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@oak//cc/crypto:signing_key",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:intrinsic",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_aggregator",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:config_converter",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:configuration_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_test(
    name = "kms_session_test",
    size = "small",
    srcs = ["kms_session_test.cc"],
    deps = [
        ":budget_cc_proto",
        ":inference_model",
        ":kms_session",
        ":private_state",
        ":range_tracker",
        "//containers:big_endian",
        "//containers:crypto_test_utils",
        "//containers/fed_sql/testing:mocks",
        "//containers/fed_sql/testing:test_utils",
        "//testing:matchers",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:int128",
        "@federated-compute//fcp/confidentialcompute:cose",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
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

cc_library(
    name = "interval_set",
    hdrs = ["interval_set.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:check",
    ],
)

cc_test(
    name = "interval_set_test",
    size = "small",
    srcs = ["interval_set_test.cc"],
    deps = [
        ":interval_set",
        "@googletest//:gtest_main",
    ],
)

proto_library(
    name = "range_tracker_proto",
    srcs = ["range_tracker.proto"],
)

# buildifier: disable=native-cc-proto
cc_proto_library(
    name = "range_tracker_cc_proto",
    deps = [":range_tracker_proto"],
)

cc_library(
    name = "range_tracker",
    srcs = ["range_tracker.cc"],
    hdrs = ["range_tracker.h"],
    deps = [
        ":interval_set",
        ":range_tracker_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
    ],
)

cc_test(
    name = "range_tracker_test",
    srcs = ["range_tracker_test.cc"],
    deps = [
        ":range_tracker",
        "//testing:matchers",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/status",
        "@googletest//:gtest_main",
    ],
)

proto_library(
    name = "budget_proto",
    srcs = ["budget.proto"],
)

# buildifier: disable=native-cc-proto
cc_proto_library(
    name = "budget_cc_proto",
    deps = [":budget_proto"],
)

cc_library(
    name = "budget",
    srcs = ["budget.cc"],
    hdrs = ["budget.h"],
    deps = [
        ":budget_cc_proto",
        ":range_tracker",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "budget_test",
    srcs = ["budget_test.cc"],
    deps = [
        ":budget",
        "//testing:matchers",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/status",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "private_state",
    hdrs = ["private_state.h"],
    deps = [
        ":budget",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":confidential_transform_server",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
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
