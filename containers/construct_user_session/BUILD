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
    name = "ingestion",
    srcs = ["ingestion.cc"],
    hdrs = ["ingestion.h"],
    deps = [
        "//containers/common:checkpoint_utils",
        "//containers/common:row_set",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@federated_compute//fcp/confidentialcompute:constants",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
    ],
)

cc_test(
    name = "ingestion_test",
    srcs = ["ingestion_test.cc"],
    deps = [
        ":ingestion",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@federated_compute//fcp/confidentialcompute:constants",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:in_memory_checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_library(
    name = "row_gather",
    srcs = ["row_gather.cc"],
    hdrs = ["row_gather.h"],
    deps = [
        ":ingestion",
        "//containers/common:row_set",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
    ],
)

cc_test(
    name = "row_gather_test",
    srcs = ["row_gather_test.cc"],
    deps = [
        ":ingestion",
        ":row_gather",
        "//containers/common:row_set",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@federated_compute//fcp/confidentialcompute:constants",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:in_memory_checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)
