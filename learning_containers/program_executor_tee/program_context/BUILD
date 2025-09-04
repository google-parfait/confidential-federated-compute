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
    name = "compilers",
    srcs = ["compilers.py"],
    deps = [
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
    ],
)

py_library(
    name = "execution_context",
    srcs = ["execution_context.py"],
    deps = [
        ":replace_data_pointers",
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:computation_delegation_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:file_info_py_pb2",
        "@federated-compute//fcp/protos/confidentialcompute:tff_config_py_pb2",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_py_pb2",
        "@pypi//googleapis_common_protos",
        "@pypi//portpicker",
    ],
)

py_test(
    name = "execution_context_test",
    srcs = ["execution_context_test.py"],
    data = ["//program_executor_tee/program_context/cc:computation_runner_binary_tensorflow"],
    tags = ["noasan"],
    deps = [
        ":compilers",
        ":execution_context",
        ":test_helpers",
        "//program_executor_tee/program_context/cc:fake_service_bindings",
        "@federated-compute//fcp/confidentialcompute/python:compiler",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//absl_py",
        "@pypi//numpy",
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
        ":test_helpers",
        "//program_executor_tee/program_context/cc:fake_service_bindings",
        "@federated_language//federated_language",
        "@federated_language_jax//federated_language_jax/computation:jax_computation",
        "@pypi//absl_py",
        "@pypi//portpicker",
    ],
)

py_library(
    name = "program_runner",
    srcs = ["program_runner.py"],
    data = ["//program_executor_tee/program_context/cc:computation_runner_binary_tensorflow"],
    visibility = ["//visibility:public"],
    deps = [
        ":compilers",
        ":execution_context",
        ":release_manager",
        "@federated-compute//fcp/confidentialcompute/python:program_input_provider",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_py_pb2",
    ],
)

py_test(
    name = "program_runner_test",
    srcs = ["program_runner_test.py"],
    tags = ["noasan"],
    deps = [
        ":program_runner",
        ":test_helpers",
        "//program_executor_tee/program_context/cc:fake_service_bindings",
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
    name = "release_manager",
    srcs = ["release_manager.py"],
    deps = [
        "@com_github_grpc_grpc//src/python/grpcio/grpc:grpcio",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
    ],
)

py_test(
    name = "release_manager_test",
    srcs = ["release_manager_test.py"],
    tags = ["noasan"],
    deps = [
        ":release_manager",
        "//program_executor_tee/program_context/cc:fake_service_bindings",
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
    name = "replace_data_pointers",
    srcs = ["replace_data_pointers.py"],
    deps = [
        "@federated-compute//fcp/protos/confidentialcompute:file_info_py_pb2",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_py_pb2",
    ],
)

py_test(
    name = "replace_data_pointers_test",
    srcs = ["replace_data_pointers_test.py"],
    deps = [
        ":replace_data_pointers",
        ":test_helpers",
        "@federated-compute//fcp/protos/confidentialcompute:file_info_py_pb2",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_py_pb2",
        "@pypi//absl_py",
        "@pypi//attrs",
        "@pypi//dm_tree",
        "@pypi//ml_dtypes",
        "@pypi//numpy",
        "@pypi//typing_extensions",
    ],
)

py_library(
    name = "test_helpers",
    testonly = True,
    srcs = ["test_helpers.py"],
    deps = [
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:file_info_py_pb2",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_py_pb2",
        "@pypi//googleapis_common_protos",
    ],
)
