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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(
    default_visibility = ["//:__subpackages__"],
    features = ["layering_check"],
)

cc_library(
    name = "sqlite_adapter",
    srcs = ["sqlite_adapter.cc"],
    hdrs = ["sqlite_adapter.h"],
    deps = [
        ":row_set",
        ":row_view",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf_lite",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/client:example_query_result_cc_proto",
        "@federated-compute//fcp/protos:data_type_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
        "@sqlite",
    ],
)

cc_test(
    name = "sqlite_adapter_test",
    size = "small",
    srcs = ["sqlite_adapter_test.cc"],
    deps = [
        ":input",
        ":sqlite_adapter",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/protos:data_type_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@sqlite",
    ],
)

cc_test(
    name = "sqlite_adapter_bm",
    size = "small",
    srcs = ["sqlite_adapter_bm.cc"],
    tags = ["manual"],
    deps = [
        ":row_set",
        ":sqlite_adapter",
        "//containers/fed_sql:session_utils",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/confidentialcompute:constants",
        "@federated-compute//fcp/protos:data_type_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@google_benchmark//:benchmark",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@sqlite",
    ],
)

cc_library(
    name = "input",
    srcs = ["input.cc"],
    hdrs = ["input.h"],
    deps = [
        ":row_view",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:variant",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
    ],
)

cc_test(
    name = "input_test",
    size = "small",
    srcs = ["input_test.cc"],
    deps = [
        ":input",
        ":row_view",
        "//testing:matchers",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_library(
    name = "row_set",
    srcs = ["row_set.cc"],
    hdrs = ["row_set.h"],
    deps = [
        ":input",
        ":row_view",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@federated-compute//fcp/base",
    ],
)

cc_test(
    name = "row_set_test",
    size = "small",
    srcs = ["row_set_test.cc"],
    deps = [
        ":input",
        ":row_set",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_library(
    name = "row_view",
    srcs = ["row_view.cc"],
    hdrs = ["row_view.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base:base",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
    ],
)

cc_test(
    name = "row_view_test",
    size = "small",
    srcs = ["row_view_test.cc"],
    deps = [
        ":row_view",
        "//testing:matchers",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)
