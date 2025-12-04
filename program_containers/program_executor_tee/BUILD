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

load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:defs.bzl", "py_library")
load("//program_executor_tee:pkg_tar_runfiles.bzl", "pkg_tar_runfiles")
load("//program_executor_tee:python_runtime.bzl", "trusted_py_runtime")

package(features = ["layering_check"])

###############################################################################
# Basic root
###############################################################################

trusted_py_runtime(
    name = "trusted_program_py_runtime",
    py_deps = [
        "//program_executor_tee/program_context:program_runner",
    ],
    source_repo_names = [
        "federated-compute",
        "org_tensorflow_federated",
        "oak",
    ],
)

cc_library(
    name = "python_manager",
    srcs = ["python_manager.cc"],
    hdrs = ["python_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@pybind11",
    ],
)

proto_library(
    name = "budget_proto",
    srcs = ["budget.proto"],
)

# buildifier: disable=native-cc-proto
cc_proto_library(
    name = "budget_cc_proto",
    deps = [":budget_proto"],
)

cc_library(
    name = "private_state",
    srcs = ["private_state.cc"],
    hdrs = ["private_state.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":budget_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "private_state_test",
    srcs = ["private_state_test.cc"],
    visibility = [":__subpackages__"],
    deps = [
        ":budget_cc_proto",
        ":private_state",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "confidential_transform_server",
    srcs = ["confidential_transform_server.cc"],
    hdrs = ["confidential_transform_server.h"],
    data = [":trusted_program_py_runtime"],
    visibility = ["//visibility:public"],
    deps = [
        ":private_state",
        ":python_manager",
        "//program_executor_tee/program_context/cc:data_parser",
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
        "@federated-compute//fcp/confidentialcompute:private_state",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:program_executor_tee_cc_proto",
        "@oak//cc/crypto:signing_key",
        "@pybind11",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_library(
    name = "testing_base",
    hdrs = ["testing_base.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":confidential_transform_server",
        ":python_manager",
        "//program_executor_tee/program_context/cc:fake_data_read_write_service",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@confidential-federated-compute//containers:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:kms_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@oak//cc/crypto:client_encryptor",
        "@oak//cc/crypto:encryption_key",
        "@oak//proto/attestation:reference_value_cc_proto",
    ],
)

cc_test(
    name = "confidential_transform_server_test",
    size = "medium",
    srcs = ["confidential_transform_server_test.cc"],
    data = [
        "testdata/test_reference_values.txtpb",
    ],
    env = {
        "PYTHONPATH": "$(PYTHONPATH)",
        "PYTHONHOME": "$(PYTHONHOME)",
    },
    tags = ["noasan"],
    toolchains = [":trusted_program_py_runtime"],
    deps = [
        ":testing_base",
        "//program_executor_tee/program_context/cc:generate_checkpoint",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/confidentialcompute:private_state",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:program_executor_tee_cc_proto",
        "@federated-compute//fcp/testing:parse_text_proto",
        "@googletest//:gtest",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_binary(
    name = "root_main_basic",
    srcs = ["root_main_basic.cc"],
    local_defines = [
        "CPP_PYTHON_PATH=\\\"$(PYTHONPATH)\\\"",
    ],
    tags = ["noasan"],
    toolchains = [":trusted_program_py_runtime"],
    deps = [
        ":confidential_transform_server",
        ":python_manager",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

pkg_tar_runfiles(
    name = "root_basic_runfiles",
    srcs = [":root_main_basic"],
    runfiles_prefix = "root_main_basic.runfiles",
)

pkg_tar(
    name = "root_basic_tar",
    srcs = [":root_basic_runfiles"],
)

oci_image(
    name = "root_basic_oci_image",
    # Use a base image that provides the same version of python we reference
    # in the WORKSPACE file.
    base = "@python_3_12_slim_bookworm",
    # Set the workdir and entrypoint so that the main binary will be executed
    # from within the runfiles directory. This ensures that the main binary
    # will be able to find its runfiles properly.
    entrypoint = ["program_executor_tee/root_main_basic"],
    tars = [":root_basic_tar"],
    workdir = "/root_main_basic.runfiles/__main__/",
)

oci_runtime_bundle(
    name = "root_basic_oci_runtime_bundle",
    image = ":root_basic_oci_image",
    visibility = ["//:__pkg__"],
)

###############################################################################
# XLA root
###############################################################################

py_library(
    name = "initialize_program_xla",
    srcs = ["initialize_program_xla.py"],
    deps = [
        "//program_executor_tee/program_context:execution_context",
        "@org_tensorflow_federated//tensorflow_federated",
    ],
)

trusted_py_runtime(
    name = "trusted_program_py_xla_runtime",
    py_deps = [
        ":initialize_program_xla",
        "//program_executor_tee/program_context:program_runner",
        "@federated_language_jax//federated_language_jax",
        "@pypi//jax_privacy",
    ],
    source_repo_names = [
        "federated_language",
        "federated_language_jax",
        "federated-compute",
        "org_tensorflow_federated",
        "oak",
    ],
)

cc_library(
    name = "confidential_transform_server_xla",
    srcs = ["confidential_transform_server_xla.cc"],
    hdrs = ["confidential_transform_server_xla.h"],
    data = [":trusted_program_py_xla_runtime"],
    deps = [
        ":confidential_transform_server",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

cc_test(
    name = "confidential_transform_server_xla_test",
    srcs = ["confidential_transform_server_xla_test.cc"],
    data = [
        "//program_executor_tee/program_context/cc:computation_runner_binary_xla",
    ],
    env = {
        "PYTHONPATH": "$(PYTHONPATH)",
        "PYTHONHOME": "$(PYTHONHOME)",
    },
    tags = ["noasan"],
    toolchains = [":trusted_program_py_xla_runtime"],
    deps = [
        ":confidential_transform_server_xla",
        ":testing_base",
        "//program_executor_tee/program_context/cc:fake_data_read_write_service",
        "//program_executor_tee/program_context/cc:generate_checkpoint",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@googletest//:gtest",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_binary(
    name = "root_main_xla",
    srcs = ["root_main_xla.cc"],
    data = [
        "//program_executor_tee/program_context/cc:computation_runner_binary_xla",
    ],
    local_defines = [
        "CPP_PYTHON_PATH=\\\"$(PYTHONPATH)\\\"",
    ],
    tags = ["noasan"],
    toolchains = [":trusted_program_py_xla_runtime"],
    deps = [
        ":confidential_transform_server_xla",
        ":python_manager",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

pkg_tar_runfiles(
    name = "root_xla_runfiles",
    srcs = [":root_main_xla"],
    runfiles_prefix = "root_main_xla.runfiles",
)

pkg_tar(
    name = "root_xla_tar",
    srcs = [":root_xla_runfiles"],
)

oci_image(
    name = "root_xla_oci_image",
    # Use a base image that provides the same version of python we reference
    # in the WORKSPACE file.
    base = "@python_3_12_slim_bookworm",
    # Set the workdir and entrypoint so that the main binary will be executed
    # from within the runfiles directory. This ensures that the main binary
    # will be able to find its runfiles properly.
    entrypoint = ["program_executor_tee/root_main_xla"],
    tars = [":root_xla_tar"],
    workdir = "/root_main_xla.runfiles/__main__/",
)

oci_runtime_bundle(
    name = "root_xla_oci_runtime_bundle",
    image = ":root_xla_oci_image",
    visibility = ["//:__pkg__"],
)

###############################################################################
# Basic worker
###############################################################################

cc_library(
    name = "program_worker_server",
    srcs = ["program_worker_server.cc"],
    hdrs = ["program_worker_server.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//program_executor_tee/program_context/cc:noise_leaf_executor",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@federated-compute//fcp/protos/confidentialcompute:program_worker_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:program_worker_cc_proto",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:server_session",
    ],
)

###############################################################################
# XLA worker
###############################################################################

cc_binary(
    name = "worker_main_xla",
    srcs = ["worker_main_xla.cc"],
    tags = ["noasan"],
    deps = [
        ":program_worker_server",
        "//program_executor_tee/program_context/cc:server_session_config",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated_language_jax//federated_language_jax/executor:xla_executor",
        "@local_xla//xla/service:cpu_plugin",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/ffi:bytes_bindings",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/ffi:error_bindings",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
        "@oak//proto/session:session_cc_proto",
    ],
)

pkg_tar_runfiles(
    name = "worker_xla_runfiles",
    srcs = [":worker_main_xla"],
    runfiles_prefix = "worker_main_xla.runfiles",
)

pkg_tar(
    name = "worker_xla_tar",
    srcs = [":worker_xla_runfiles"],
)

oci_image(
    name = "worker_xla_oci_image",
    base = "@distroless_cc_debian12",
    entrypoint = ["/main"],
    tars = [":worker_xla_tar"],
)

oci_runtime_bundle(
    name = "worker_xla_oci_runtime_bundle",
    image = ":worker_xla_oci_image",
    visibility = ["//:__pkg__"],
)
