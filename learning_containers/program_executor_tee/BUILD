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
load("//program_executor_tee:pkg_tar_runfiles.bzl", "pkg_tar_runfiles")
load("//program_executor_tee:python_runtime.bzl", "trusted_py_runtime")

package(features = ["layering_check"])

trusted_py_runtime(
    name = "trusted_program_py_runtime",
    py_deps = [
        "//program_executor_tee/program_context:program_runner",
    ],
    source_repo_names = [
        "federated_language",
        "federated-compute",
        "org_tensorflow_federated",
        "oak",
    ],
)

filegroup(
    name = "testdata",
    srcs = glob(["testdata/*"]),
)

cc_library(
    name = "testing_base",
    hdrs = ["testing_base.h"],
    data = [":testdata"],
    deps = [
        ":confidential_transform_server",
        "//program_executor_tee/program_context/cc:fake_data_read_write_service",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
    # Include py runfiles needed to execute the federated program.
    data = [":trusted_program_py_runtime"],
    deps = [
        "//program_executor_tee/program_context/cc:data_parser",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
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

cc_test(
    name = "data_source_integration_test",
    size = "medium",
    srcs = ["data_source_integration_test.cc"],
    env = {
        "PYTHONPATH": "$(PYTHONPATH)",
        "PYTHONHOME": "$(PYTHONHOME)",
    },
    tags = ["noasan"],
    toolchains = [":trusted_program_py_runtime"],
    deps = [
        ":testing_base",
        "//program_executor_tee/program_context/cc:fake_data_read_write_service",
        "//program_executor_tee/program_context/cc:generate_checkpoint",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_test(
    name = "load_model_integration_test",
    size = "medium",
    srcs = ["load_model_integration_test.cc"],
    env = {
        "PYTHONPATH": "$(PYTHONPATH)",
        "PYTHONHOME": "$(PYTHONHOME)",
    },
    tags = ["noasan"],
    toolchains = [":trusted_program_py_runtime"],
    deps = [
        ":testing_base",
        "//program_executor_tee/program_context/cc:fake_data_read_write_service",
        "//program_executor_tee/program_context/cc:generate_checkpoint",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    local_defines = [
        "CPP_PYTHON_PATH=\\\"$(PYTHONPATH)\\\"",
    ],
    tags = ["noasan"],
    toolchains = [":trusted_program_py_runtime"],
    deps = [
        ":confidential_transform_server",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

pkg_tar_runfiles(
    name = "runfiles",
    srcs = [":main"],
    runfiles_prefix = "main.runfiles",
)

pkg_tar(
    name = "tar",
    srcs = [":runfiles"],
)

oci_image(
    name = "oci_image",
    # Use a base image that provides the same version of python we reference
    # in the WORKSPACE file.
    base = "@python_3_10_slim_bookworm",
    # Set the workdir and entrypoint so that the main binary will be executed
    # from within the runfiles directory. This ensures that the main binary
    # will be able to find its runfiles properly.
    entrypoint = ["program_executor_tee/main"],
    tars = [":tar"],
    workdir = "/main.runfiles/__main__/",
)

oci_runtime_bundle(
    name = "oci_runtime_bundle",
    image = ":oci_image",
    visibility = ["//:__pkg__"],
)
