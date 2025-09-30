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
load("@rules_python//python:defs.bzl", "py_library")
load("//program_executor_tee:python_runtime.bzl", "trusted_py_runtime")

package(features = ["layering_check"])

trusted_py_runtime(
    name = "trusted_program_py_runtime",
    py_deps = [
        ":initialize_program_xla",
        "//program_executor_tee/program_context:program_runner",
        "@federated-compute//fcp/confidentialcompute/python:min_sep_data_source",
        "@federated_language//federated_language",
        "@federated_language_jax//federated_language_jax",
    ],
    source_repo_names = [
        "federated_language",
        "federated_language_jax",
        "federated-compute",
        "org_tensorflow_federated",
        "oak",
    ],
)

cc_test(
    name = "computation_integration_test",
    srcs = ["computation_integration_test.cc"],
    data = [
        "//program_executor_tee/program_context/cc:computation_runner_binary_xla",
    ],
    env = {
        "PYTHONPATH": "$(PYTHONPATH)",
        "PYTHONHOME": "$(PYTHONHOME)",
    },
    tags = ["noasan"],
    toolchains = [":trusted_program_py_runtime"],
    deps = [
        ":confidential_transform_server_xla",
        ":testing_base",
        "//program_executor_tee/program_context/cc:fake_data_read_write_service",
        "//program_executor_tee/program_context/cc:generate_checkpoint",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_library(
    name = "confidential_transform_server_xla",
    srcs = ["confidential_transform_server_xla.cc"],
    hdrs = ["confidential_transform_server_xla.h"],
    data = [
        ":trusted_program_py_runtime",
    ],
    deps = [
        ":confidential_transform_server",
        "//program_executor_tee/program_context/cc:data_parser",
        "@com_google_absl//absl/strings",
        "@oak//cc/crypto:signing_key",
        "@pybind11",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_library(
    name = "initialize_program_xla",
    srcs = ["initialize_program_xla.py"],
    deps = [
        "//program_executor_tee/program_context:execution_context",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_py_pb2",
    ],
)

cc_library(
    name = "testing_base",
    hdrs = ["testing_base.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":confidential_transform_server",
        "//program_executor_tee/program_context/cc:fake_data_read_write_service",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@confidential-federated-compute//containers:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "confidential_transform_server",
    srcs = ["confidential_transform_server.cc"],
    hdrs = ["confidential_transform_server.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:protobuf",
        "@confidential-federated-compute//containers:blob_metadata",
        "@confidential-federated-compute//containers:confidential_transform_server_base",
        "@confidential-federated-compute//containers:crypto",
        "@confidential-federated-compute//containers:session",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:program_executor_tee_cc_proto",
        "@oak//cc/crypto:signing_key",
        "@pybind11",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_test(
    name = "confidential_transform_server_test",
    size = "medium",
    srcs = ["confidential_transform_server_test.cc"],
    env = {
        "PYTHONPATH": "$(PYTHONPATH)",
        "PYTHONHOME": "$(PYTHONHOME)",
    },
    tags = ["noasan"],
    toolchains = [":trusted_program_py_runtime"],
    deps = [
        ":testing_base",
        "@com_github_grpc_grpc//:grpc++",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:program_executor_tee_cc_proto",
        "@federated-compute//fcp/testing:parse_text_proto",
        "@googletest//:gtest",
    ],
)
