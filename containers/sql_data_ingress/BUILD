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

load("@oak//oak_containers/app_base:defs.bzl", "app_bundle")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(features = ["layering_check"])

cc_library(
    name = "sql_utils",
    srcs = ["sql_utils.cc"],
    hdrs = ["sql_utils.h"],
    deps = [
        "//containers/sql:row_set",
        "//containers/sql:sqlite_adapter",
        "@com_google_absl//absl/status:statusor",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
    ],
)

cc_test(
    name = "sql_utils_test",
    srcs = ["sql_utils_test.cc"],
    deps = [
        ":sql_utils",
        "//containers/sql:input",
        "//containers/sql:row_set",
        "//containers/sql:sqlite_adapter",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_library(
    name = "sql_data_ingress_fn",
    srcs = ["sql_data_ingress_fn.cc"],
    hdrs = ["sql_data_ingress_fn.h"],
    deps = [
        ":sql_utils",
        "//containers:session",
        "//containers/fns",
        "//containers/sql:input",
        "//containers/sql:row_set",
        "//containers/sql:sqlite_adapter",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_data_ingress_config_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_test(
    name = "sql_data_ingress_fn_test",
    srcs = ["sql_data_ingress_fn_test.cc"],
    deps = [
        ":sql_data_ingress_fn",
        "//containers:session",
        "//containers/fns",
        "//containers/testing",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_protobuf//:any_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_data_ingress_config_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":sql_data_ingress_fn",
        "//containers/fns",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

app_bundle(
    name = "app_bundle",
    binary = ":main",
    visibility = ["//:__pkg__"],
)
