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

load("@oak//oak_containers/app_base:defs.bzl", "app_bundle")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
)

cc_library(
    name = "batched_inference_engine",
    srcs = [],
    hdrs = ["batched_inference_engine.h"],
    deps = ["@com_google_absl//absl/status:statusor"],
)

cc_library(
    name = "batched_inference_fn",
    srcs = ["batched_inference_fn.cc"],
    hdrs = ["batched_inference_fn.h"],
    deps = [
        ":batched_inference_engine",
        "//containers/fed_sql:inference_model_helper",
        "//containers/fed_sql:session_utils",
        "//containers/fns",
        "//containers/sql:input",
        "//containers/sql:row_set",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:any_cc_proto",
        "@federated-compute//fcp/confidentialcompute:time_window_utilities",
        "@federated-compute//fcp/protos:data_type_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_library(
    name = "batched_inference_test_utils",
    testonly = True,
    srcs = ["batched_inference_test_utils.cc"],
    hdrs = ["batched_inference_test_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:any_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
    ],
)

cc_test(
    name = "batched_inference_fn_test",
    srcs = ["batched_inference_fn_test.cc"],
    deps = [
        ":batched_inference_fn",
        ":batched_inference_test_utils",
        "//containers:session",
        "//containers/fns",
        "//containers/testing",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:any_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "batched_inference_server",
    srcs = [
        "batched_inference_server.cc",
    ],
    hdrs = [
        "batched_inference_server.h",
    ],
    deps = [
        ":batched_inference_engine",
        ":batched_inference_fn",
        "//containers/fns",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:any_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

cc_test(
    name = "batched_inference_server_test",
    srcs = ["batched_inference_server_test.cc"],
    defines = ["FCP_CLIENT_SUPPORT_CONFIDENTIAL_AGG"],
    deps = [
        ":batched_inference_engine",
        ":batched_inference_server",
        ":batched_inference_test_utils",
        "//containers:crypto_test_utils",
        "//testing:parse_text_proto",
        "@boringssl//:crypto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googleapis//google/rpc:code_cc_proto",
        "@com_google_protobuf//:any_cc_proto",
        "@com_google_protobuf//:struct_cc_proto",
        "@federated-compute//fcp/base:compression",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/confidentialcompute:constants",
        "@federated-compute//fcp/confidentialcompute:cose",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:kms_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@oak//cc/crypto:client_encryptor",
        "@oak//cc/crypto:common",
        "@oak//cc/crypto:encryption_key",
        "@oak//proto/crypto:crypto_cc_proto",
    ],
)

cc_binary(
    name = "main_for_testing",
    srcs = ["main_for_testing.cc"],
    deps = [
        ":batched_inference_engine",
        ":batched_inference_server",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

app_bundle(
    name = "app_bundle_for_testing",
    binary = ":main_for_testing",
)
