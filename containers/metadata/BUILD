# Copyright 2025 Google LLC.
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
    name = "confidential_transform_server",
    srcs = ["confidential_transform_server.cc"],
    hdrs = ["confidential_transform_server.h"],
    deps = [
        "//containers:confidential_transform_server_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:any_cc_proto",
        "@com_google_protobuf//:struct_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tee_payload_metadata_cc_proto",
        "@oak//cc/crypto:encryption_key",
        "@oak//cc/crypto:signing_key",
    ],
)

cc_test(
    name = "confidential_transform_server_test",
    srcs = ["confidential_transform_server_test.cc"],
    deps = [
        ":confidential_transform_server",
        "//containers:crypto_test_utils",
        "//testing:parse_text_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googleapis//google/rpc:code_cc_proto",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/confidentialcompute:constants",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:kms_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tee_payload_metadata_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@oak//cc/crypto:client_encryptor",
        "@oak//cc/crypto:encryption_key",
    ],
)

cc_library(
    name = "metadata_map_fn",
    srcs = ["metadata_map_fn.cc"],
    hdrs = ["metadata_map_fn.h"],
    deps = [
        "//containers:big_endian",
        "//containers/fns:map_fn",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@federated-compute//fcp/base:base",
        "@federated-compute//fcp/base:digest",
        "@federated-compute//fcp/confidentialcompute:constants",
        "@federated-compute//fcp/confidentialcompute:time_window_utilities",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tee_payload_metadata_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_test(
    name = "metadata_map_fn_test",
    srcs = ["metadata_map_fn_test.cc"],
    deps = [
        ":metadata_map_fn",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googleapis//google/rpc:code_cc_proto",
        "@federated-compute//fcp/confidentialcompute:constants",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tee_payload_metadata_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)
