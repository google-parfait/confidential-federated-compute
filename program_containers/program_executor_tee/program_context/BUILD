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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(features = ["layering_check"])

py_library(
    name = "execution_context",
    srcs = ["execution_context.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_config_py_pb2",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_py_pb2",
        "@pypi//googleapis_common_protos",
        "@pypi//portpicker",
    ],
)

py_test(
    name = "execution_context_jax_test",
    srcs = ["execution_context_jax_test.py"],
    data = ["//program_executor_tee/program_context/cc:computation_runner_binary_xla"],
    tags = ["noasan"],
    deps = [
        ":execution_context",
        "//program_executor_tee/program_context/cc:fake_service_bindings_jax",
        "@federated_language//federated_language",
        "@federated_language_jax//federated_language_jax/computation:jax_computation",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//absl_py",
        "@pypi//parameterized",
        "@pypi//portpicker",
    ],
)

py_library(
    name = "program_runner",
    srcs = ["program_runner.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":external_service_handle",
        ":external_service_handle_ledger",
        "@federated-compute//fcp/confidentialcompute/python:program_input_provider",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor_py_pb2",
    ],
)

py_test(
    name = "program_runner_test",
    srcs = ["program_runner_test.py"],
    tags = ["noasan"],
    deps = [
        ":execution_context",
        ":program_runner",
        "//program_executor_tee/program_context/cc:fake_service_bindings_jax",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//absl_py",
        "@pypi//numpy",
        "@pypi//portpicker",
    ],
)

py_library(
    name = "external_service_handle_ledger",
    srcs = ["external_service_handle_ledger.py"],
    deps = [
        "@federated-compute//fcp/confidentialcompute/python:external_service_handle",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
        "@pypi//googleapis_common_protos",
    ],
)

py_test(
    name = "external_service_handle_ledger_test",
    srcs = ["external_service_handle_ledger_test.py"],
    tags = ["noasan"],
    deps = [
        ":external_service_handle_ledger",
        "//program_executor_tee/program_context/cc:fake_service_bindings_jax",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//absl_py",
        "@pypi//numpy",
        "@pypi//portpicker",
    ],
)

py_library(
    name = "external_service_handle",
    srcs = ["external_service_handle.py"],
    deps = [
        "@federated-compute//fcp/confidentialcompute/python:external_service_handle",
    ],
)

py_test(
    name = "external_service_handle_test",
    srcs = ["external_service_handle_test.py"],
    tags = ["noasan"],
    deps = [
        ":external_service_handle",
        "@pypi//absl_py",
    ],
)
