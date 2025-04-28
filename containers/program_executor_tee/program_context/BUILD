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

py_library(
    name = "compilers",
    srcs = ["compilers.py"],
    deps = [
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//absl_py",
        "@pypi//attrs",
        "@pypi//dm_tree",
        "@pypi//google_vizier",
        "@pypi//jax",
        "@pypi//jaxlib",
        "@pypi//ml_dtypes",
        "@pypi//portpicker",
        "@pypi//tensorflow",
        "@pypi//tensorflow_model_optimization",
        "@pypi//tensorflow_privacy",
        "@pypi//typing_extensions",
    ],
)

py_library(
    name = "computation_runner_bindings",
    data = ["//containers/program_executor_tee/program_context/cc:computation_runner_bindings.so"],
)

py_library(
    name = "fake_data_read_write_servicer",
    srcs = ["fake_data_read_write_servicer.py"],
    deps = [
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
    ],
)

py_library(
    name = "execution_context",
    srcs = ["execution_context.py"],
    deps = [
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//absl_py",
        "@pypi//attrs",
        "@pypi//dm_tree",
        "@pypi//google_vizier",
        "@pypi//jax",
        "@pypi//jaxlib",
        "@pypi//ml_dtypes",
        "@pypi//portpicker",
        "@pypi//tensorflow",
        "@pypi//tensorflow_model_optimization",
        "@pypi//tensorflow_privacy",
        "@pypi//typing_extensions",
    ],
)

py_test(
    name = "execution_context_test",
    srcs = ["execution_context_test.py"],
    deps = [
        ":compilers",
        ":computation_runner_bindings",
        ":execution_context",
        "@federated_language//federated_language",
        "@pypi//absl_py",
        "@pypi//attrs",
        "@pypi//dm_tree",
        "@pypi//ml_dtypes",
        "@pypi//typing_extensions",
    ],
)

py_library(
    name = "release_manager",
    srcs = ["release_manager.py"],
    deps = [
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//absl_py",
        "@pypi//attrs",
        "@pypi//dm_tree",
        "@pypi//google_vizier",
        "@pypi//jax",
        "@pypi//jaxlib",
        "@pypi//ml_dtypes",
        "@pypi//portpicker",
        "@pypi//tensorflow",
        "@pypi//tensorflow_model_optimization",
        "@pypi//tensorflow_privacy",
        "@pypi//typing_extensions",
    ],
)

py_test(
    name = "release_manager_test",
    srcs = ["release_manager_test.py"],
    deps = [
        ":fake_data_read_write_servicer",
        ":release_manager",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//absl_py",
        "@pypi//attrs",
        "@pypi//dm_tree",
        "@pypi//google_vizier",
        "@pypi//jax",
        "@pypi//jaxlib",
        "@pypi//ml_dtypes",
        "@pypi//portpicker",
        "@pypi//tensorflow",
        "@pypi//tensorflow_model_optimization",
        "@pypi//tensorflow_privacy",
        "@pypi//typing_extensions",
    ],
)

py_library(
    name = "program_runner",
    srcs = ["program_runner.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":compilers",
        ":computation_runner_bindings",
        ":execution_context",
        ":release_manager",
        "@federated_language//federated_language",
        "@pypi//absl_py",
        "@pypi//attrs",
        "@pypi//dm_tree",
        "@pypi//ml_dtypes",
        "@pypi//typing_extensions",
    ],
)

py_test(
    name = "program_runner_test",
    srcs = ["program_runner_test.py"],
    deps = [
        ":fake_data_read_write_servicer",
        ":program_runner",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_py_proto",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//absl_py",
        "@pypi//attrs",
        "@pypi//dm_tree",
        "@pypi//google_vizier",
        "@pypi//jax",
        "@pypi//jaxlib",
        "@pypi//ml_dtypes",
        "@pypi//portpicker",
        "@pypi//tensorflow",
        "@pypi//tensorflow_model_optimization",
        "@pypi//tensorflow_privacy",
        "@pypi//typing_extensions",
    ],
)
