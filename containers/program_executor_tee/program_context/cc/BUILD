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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//containers/program_executor_tee/program_context:__subpackages__"])

cc_library(
    name = "computation_runner",
    srcs = ["computation_runner.cc"],
    hdrs = ["computation_runner.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/confidentialcompute:tff_execution_helper",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
    ],
)

pybind_extension(
    name = "computation_runner_bindings",
    srcs = ["computation_runner_bindings.cc"],
    deps = [
        ":computation_runner",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

cc_library(
    name = "data_uri_resolver",
    srcs = ["data_uri_resolver.cc"],
    hdrs = ["data_uri_resolver.h"],
    deps = [
        "//containers:crypto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:file_info_cc_proto",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/tensorflow:converters",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:data_backend",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:tensor_serialization",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:tensorflow_utils",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_test(
    name = "data_uri_resolver_test",
    srcs = ["data_uri_resolver_test.cc"],
    deps = [
        ":data_uri_resolver",
        ":fake_data_read_write_service",
        "//testing:matchers",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@federated-compute//fcp/protos/confidentialcompute:file_info_cc_proto",
        "@googletest//:gtest_main",
        "@oak//proto/containers:orchestrator_crypto_cc_grpc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:tensor_serialization",
    ],
)

cc_library(
    name = "fake_data_read_write_service",
    testonly = True,
    srcs = ["fake_data_read_write_service.cc"],
    hdrs = ["fake_data_read_write_service.h"],
    visibility = ["//containers/program_executor_tee:__subpackages__"],
    deps = [
        "//containers:blob_metadata",
        "//containers:crypto_test_utils",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
    ],
)

cc_test(
    name = "fake_data_read_write_service_test",
    srcs = ["fake_data_read_write_service_test.cc"],
    deps = [
        ":fake_data_read_write_service",
        "//containers:crypto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@googletest//:gtest_main",
        "@oak//proto/containers:orchestrator_crypto_cc_grpc",
    ],
)

cc_library(
    name = "noise_client_session",
    srcs = ["noise_client_session.cc"],
    hdrs = ["noise_client_session.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_proto",
        "@oak//cc/oak_session:client_session",
        "@oak//cc/oak_session:config",
        "@oak//proto/session:session_cc_proto",
    ],
)

cc_test(
    name = "noise_client_session_test",
    srcs = ["noise_client_session_test.cc"],
    deps = [
        ":noise_client_session",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_proto",
        "@googletest//:gtest_main",
        "@oak//cc/ffi:bytes_bindings",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/ffi:error_bindings",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
        "@oak//cc/oak_session:server_session",
    ],
)

cc_library(
    name = "mock_noise_client_session",
    testonly = True,
    hdrs = ["mock_noise_client_session.h"],
    deps = [
        ":noise_client_session",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest_main",
        "@oak//cc/oak_session:config",
        "@oak//proto/session:session_cc_proto",
    ],
)

cc_library(
    name = "computation_delegation_lambda_runner",
    srcs = ["computation_delegation_lambda_runner.cc"],
    hdrs = ["computation_delegation_lambda_runner.h"],
    deps = [
        ":noise_client_session",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/confidentialcompute:lambda_runner",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:tff_config_cc_proto",
        "@oak//proto/session:session_cc_proto",
    ],
)

cc_test(
    name = "computation_delegation_lambda_runner_test",
    srcs = ["computation_delegation_lambda_runner_test.cc"],
    deps = [
        ":computation_delegation_lambda_runner",
        ":mock_noise_client_session",
        "//testing:matchers",
        "//testing:parse_text_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:tff_config_cc_proto",
        "@googletest//:gtest_main",
        "@oak//cc/ffi:bytes_bindings",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/ffi:error_bindings",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
    ],
)
