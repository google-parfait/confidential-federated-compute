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
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_rust//rust:defs.bzl", "rust_library")

package(
    default_visibility = ["//program_executor_tee:__subpackages__"],
    features = ["layering_check"],
)

cc_library(
    name = "computation_runner",
    srcs = ["computation_runner.cc"],
    hdrs = ["computation_runner.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":client_session_config",
        ":noise_client_session",
        ":noise_executor_stub",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/confidentialcompute:composing_tee_executor",
        "@federated-compute//fcp/confidentialcompute:lambda_runner",
        "@federated-compute//fcp/confidentialcompute:tee_executor",
        "@federated-compute//fcp/confidentialcompute:tff_execution_helper",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_config_cc_proto",
        "@oak//cc/ffi:bytes_bindings",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/ffi:error_bindings",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:composing_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:streaming_remote_executor",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_binary(
    name = "computation_runner_binary_xla",
    srcs = ["computation_runner_main_xla.cc"],
    deps = [
        ":computation_runner",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@federated_language_jax//federated_language_jax/executor:xla_executor",
        "@local_xla//xla/service:cpu_plugin",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@oak//proto/attestation:reference_value_cc_proto",
    ],
)

cc_library(
    name = "data_parser",
    srcs = ["data_parser.cc"],
    hdrs = ["data_parser.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@confidential-federated-compute//containers:crypto",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:random_token",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_test(
    name = "data_parser_test",
    srcs = ["data_parser_test.cc"],
    deps = [
        ":data_parser",
        ":fake_data_read_write_service",
        ":generate_checkpoint",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@confidential-federated-compute//containers:crypto",
        "@confidential-federated-compute//containers:crypto_test_utils",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
    ],
)

cc_library(
    name = "fake_data_read_write_service",
    srcs = ["fake_data_read_write_service.cc"],
    hdrs = ["fake_data_read_write_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@confidential-federated-compute//containers:crypto_test_utils",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
    ],
)

cc_test(
    name = "fake_data_read_write_service_test",
    srcs = ["fake_data_read_write_service_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":fake_data_read_write_service",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@confidential-federated-compute//containers:crypto",
        "@confidential-federated-compute//containers:crypto_test_utils",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "generate_checkpoint",
    srcs = ["generate_checkpoint.cc"],
    hdrs = ["generate_checkpoint.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
    ],
)

cc_test(
    name = "generate_checkpoint_test",
    srcs = ["generate_checkpoint_test.cc"],
    deps = [
        ":generate_checkpoint",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_library(
    name = "noise_client_session",
    srcs = ["noise_client_session.cc"],
    hdrs = ["noise_client_session.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
        "@googletest//:gtest",
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
    name = "noise_executor_stub",
    srcs = ["noise_executor_stub.cc"],
    hdrs = ["noise_executor_stub.h"],
    deps = [
        "//program_executor_tee/program_context/cc:noise_client_session",
        "//program_executor_tee/proto:executor_wrapper_cc_proto",
        "@federated-compute//fcp/base:status_converters",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_grpc_proto",
    ],
)

cc_test(
    name = "noise_executor_stub_test",
    srcs = ["noise_executor_stub_test.cc"],
    tags = ["noasan"],
    deps = [
        ":fake_computation_delegation_service",
        ":noise_client_session",
        ":noise_executor_stub",
        ":test_helpers",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@federated_language_jax//federated_language_jax/executor:xla_executor",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@local_xla//xla/service:cpu_plugin",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/oak_session:client_session",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor",
    ],
)

cc_library(
    name = "noise_leaf_executor",
    srcs = ["noise_leaf_executor.cc"],
    hdrs = ["noise_leaf_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//program_executor_tee/proto:executor_wrapper_cc_proto",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:program_worker_cc_proto",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:server_session",
        "@oak//proto/session:session_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor_service",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_test(
    name = "noise_leaf_executor_test",
    srcs = ["noise_leaf_executor_test.cc"],
    tags = ["noasan"],
    deps = [
        ":noise_leaf_executor",
        ":server_session_config",
        ":test_helpers",
        "//program_executor_tee/proto:executor_wrapper_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@federated-compute//fcp/base:status_converters",
        "@federated_language_jax//federated_language_jax/executor:xla_executor",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@local_xla//xla/service:cpu_plugin",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/oak_session:client_session",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor",
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
        "@googletest//:gtest",
        "@oak//cc/oak_session:config",
        "@oak//proto/session:session_cc_proto",
    ],
)

cc_library(
    name = "fake_computation_delegation_service",
    srcs = ["fake_computation_delegation_service.cc"],
    hdrs = ["fake_computation_delegation_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":noise_leaf_executor",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:program_worker_cc_proto",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@oak//cc/ffi:bytes_bindings",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/ffi:error_bindings",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
        "@oak//cc/oak_session:server_session",
        "@oak//proto/session:session_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_test(
    name = "fake_computation_delegation_service_test",
    srcs = ["fake_computation_delegation_service_test.cc"],
    tags = ["noasan"],
    deps = [
        ":fake_computation_delegation_service",
        "//program_executor_tee/proto:executor_wrapper_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:program_worker_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:program_worker_cc_proto",
        "@federated_language_jax//federated_language_jax/executor:xla_executor",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@local_xla//xla/service:cpu_plugin",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/oak_session:client_session",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
    ],
)

cc_library(
    name = "fake_server",
    hdrs = ["fake_server.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":fake_computation_delegation_service",
        ":fake_data_read_write_service",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

pybind_extension(
    name = "fake_service_bindings_jax",
    srcs = ["fake_service_bindings_jax.cc"],
    deps = [
        ":fake_computation_delegation_service",
        ":fake_data_read_write_service",
        ":fake_server",
        "@federated_language_jax//federated_language_jax/executor:xla_executor",
        "@local_xla//xla/service:cpu_plugin",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

rust_library(
    name = "client_session_config",
    srcs = ["client_session_config.rs"],
    deps = [
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification",
        "@oak//oak_attestation_verification/dice_attestation_verification:oak_dice_attestation_verifier",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak//oak_time:oak_time_std",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:prost",
    ],
)

rust_library(
    name = "server_session_config",
    srcs = ["server_session_config.rs"],
    visibility = ["//visibility:public"],
    deps = [
        "@oak//oak_attestation_types",
        "@oak//oak_attestation_verification/dice_attestation_verification:oak_dice_attestation_verifier",
        "@oak//oak_attestation_verification_types",
        "@oak//oak_crypto",
        "@oak//oak_proto_rust",
        "@oak//oak_sdk/common:oak_sdk_common",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak//oak_session",
        "@oak_crates_index//:once_cell",
        "@oak_crates_index//:tokio",
    ],
)

cc_library(
    name = "test_helpers",
    srcs = ["test_helpers.cc"],
    hdrs = ["test_helpers.h"],
    deps = [
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)
