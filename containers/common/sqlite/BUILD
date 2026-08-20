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
        "//containers/common/io/tabular:row_set",
        "//containers/common/io/tabular:row_view",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf_lite",
        "@federated_compute//fcp/base:status_converters",
        "@federated_compute//fcp/client:example_query_result_cc_proto",
        "@federated_compute//fcp/protos:data_type_cc_proto",
        "@federated_compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
        "@sqlite3",
    ],
)

cc_test(
    name = "sqlite_adapter_test",
    size = "small",
    srcs = ["sqlite_adapter_test.cc"],
    deps = [
        ":sqlite_adapter",
        "//containers/common/io/tabular:input",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
        "@federated_compute//fcp/protos:data_type_cc_proto",
        "@federated_compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@sqlite3",
    ],
)

cc_test(
    name = "sqlite_adapter_bm",
    size = "small",
    srcs = ["sqlite_adapter_bm.cc"],
    tags = ["manual"],
    deps = [
        ":sqlite_adapter",
        "//containers/common/io/tabular:row_set",
        "//containers/fed_sql:session_utils",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
        "@federated_compute//fcp/confidentialcompute:constants",
        "@federated_compute//fcp/protos:data_type_cc_proto",
        "@federated_compute//fcp/protos/confidentialcompute:sql_query_cc_proto",
        "@google_benchmark//:benchmark",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@sqlite3",
    ],
)
