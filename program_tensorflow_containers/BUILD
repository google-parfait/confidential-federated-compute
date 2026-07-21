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

load("@bazel_toolchains//rules/exec_properties:exec_properties.bzl", "create_rbe_exec_properties_dict")
load("@confidential_federated_compute_program//program_executor_tee:pkg_tar_runfiles.bzl", "pkg_tar_runfiles")
load("@confidential_federated_compute_program//program_executor_tee:python_runtime.bzl", "trusted_py_runtime")
load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:install.bzl", "pkg_install")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(features = ["layering_check"])

platform(
    name = "remote_platform",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    exec_properties = create_rbe_exec_properties_dict(
        container_image = "docker://gcr.io/tensorflow-testing/nosla-cuda12.3-cudnn9.1-ubuntu20.04-manylinux2014-multipython@sha256:89aa860228092d78dc336545a5ad32e342d656a8258186327c989bd301bb3eb8",
        os_family = "Linux",
    ),
    parents = ["@local_config_platform//:host"],
)

# This rule adds a convenient way to update the requirements file.
compile_pip_requirements(
    name = "requirements",
    timeout = "long",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)

py_library(
    name = "compilers",
    srcs = ["compilers.py"],
    deps = [
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//tensorflow_cpu",
    ],
)

cc_binary(
    name = "computation_runner_binary_tensorflow",
    srcs = ["computation_runner_main_tensorflow.cc"],
    linkopts = ["-Wl,-s"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@confidential_federated_compute_program//program_executor_tee/program_context/cc:computation_runner",
        "@oak//proto/attestation:reference_value_cc_proto",
        "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
        "@org_tensorflow//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "@org_tensorflow//tensorflow/compiler/tf2xla/ops:xla_ops",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
    ],
)

pybind_extension(
    name = "fake_service_bindings_tensorflow",
    srcs = ["fake_service_bindings_tensorflow.cc"],
    deps = [
        "@confidential_federated_compute_program//program_executor_tee/program_context/cc:fake_computation_delegation_service",
        "@confidential_federated_compute_program//program_executor_tee/program_context/cc:fake_data_read_write_service",
        "@confidential_federated_compute_program//program_executor_tee/program_context/cc:fake_server",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_test(
    name = "execution_context_tensorflow_test",
    srcs = ["execution_context_tensorflow_test.py"],
    data = [":computation_runner_binary_tensorflow"],
    tags = ["noasan"],
    deps = [
        ":compilers",
        ":fake_service_bindings_tensorflow",
        "@confidential_federated_compute_program//program_executor_tee/program_context:execution_context",
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//absl_py",
        "@pypi//numpy",
        "@pypi//portpicker",
    ],
)

py_library(
    name = "initialize_program_tensorflow",
    srcs = ["initialize_program_tensorflow.py"],
    deps = [
        ":compilers",
        "@confidential_federated_compute_program//program_executor_tee/program_context:execution_context",
        "@jax_privacy//jax_privacy",
    ],
)

trusted_py_runtime(
    name = "trusted_program_py_runtime",
    py_deps = [
        "@confidential_federated_compute_program//program_executor_tee/program_context:program_runner",
        ":initialize_program_tensorflow",
        "@dpsynth//dpsynth",
        "@federated_compute//fcp/confidentialcompute/python:accounting",
        "@federated_compute//fcp/confidentialcompute/python:dp_mf_aggregator",
        "@federated_compute//fcp/confidentialcompute/python:min_sep_data_source",
        "@pypi//jax",
        "@pypi//jaxlib",
        "@pypi//optax",
    ],
    source_repo_names = [
        "confidential_federated_compute_program",
        "dpsynth",
        "federated_language",
        "federated_compute",
        "jax_privacy",
        "org_tensorflow_federated",
        "oak",
    ],
)

cc_library(
    name = "confidential_transform_server",
    srcs = ["confidential_transform_server.cc"],
    hdrs = ["confidential_transform_server.h"],
    data = [
        ":computation_runner_binary_tensorflow",
        ":trusted_program_py_runtime",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@confidential_federated_compute_program//program_executor_tee:confidential_transform_server",
        "@oak//cc/crypto:signing_key",
        "@pybind11",
    ],
)

filegroup(
    name = "testdata",
    srcs = glob(["testdata/*"]),
)

cc_test(
    name = "confidential_transform_server_test",
    size = "medium",
    srcs = ["confidential_transform_server_test.cc"],
    additional_linker_inputs = [":hide_symbols.lds"],
    data = [":testdata"],
    env = {
        "PYTHONPATH": "$(PYTHONPATH)",
        "PYTHONHOME": "$(PYTHONHOME)",
    },
    linkopts = ["-Wl,--version-script=$(location :hide_symbols.lds)"],
    linkstatic = True,
    tags = ["noasan"],
    toolchains = [":trusted_program_py_runtime"],
    deps = [
        ":confidential_transform_server",
        "@confidential_federated_compute_program//program_executor_tee:testing_base",
        "@confidential_federated_compute_program//program_executor_tee/program_context/cc:fake_data_read_write_service",
        "@confidential_federated_compute_program//program_executor_tee/program_context/cc:generate_checkpoint",
        "@federated_compute//fcp/base:random_token",
        "@federated_compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated_compute//fcp/protos/confidentialcompute:data_read_write_cc_proto",
        "@googletest//:gtest",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_binary(
    name = "libz.so.1",
    linkshared = 1,
    deps = ["@zlib"],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    additional_linker_inputs = [":hide_symbols.lds"],
    data = [
        ":libz.so.1",
    ],
    linkopts = [
        "-Wl,-s",
        "-Wl,--version-script=$(location :hide_symbols.lds)",
        # Bazel symlinks won't make it into the container tarball, so we need
        # to include the path of libpython since the Bazel _solib symlink will
        # be missing. Note: $(PYTHONHOME) ends in a slash, and the linker does
        # not collapse double slashes.
        "-Wl,-rpath,/main.runfiles/__main__/$(PYTHONHOME)lib",
        # Additionally we need the search path for `libz`, required by `numpy`,
        # which is at the toplevel due to the `data` dep above.
        "-Wl,-rpath,/main.runfiles/__main__",
        # Enable transitive lookups (e.g. libz required by numpy) to search the
        # rpath.
        "-Wl,--disable-new-dtags",
    ],
    linkstatic = True,
    local_defines = [
        "CPP_PYTHON_PATH=\\\"$(PYTHONPATH)\\\"",
        "CPP_PYTHON_HOME=\\\"$(PYTHONHOME)\\\"",
    ],
    tags = ["noasan"],
    toolchains = [":trusted_program_py_runtime"],
    deps = [
        ":confidential_transform_server",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@confidential_federated_compute_program//program_executor_tee:python_manager",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

cc_binary(
    name = "worker_main_tensorflow",
    srcs = ["worker_main_tensorflow.cc"],
    tags = ["noasan"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@confidential_federated_compute_program//program_executor_tee:program_worker_server",
        "@confidential_federated_compute_program//program_executor_tee/program_context/cc:server_session_config",
        "@federated_compute//fcp/protos/confidentialcompute:program_worker_tee_config_cc_proto",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/ffi:bytes_bindings",
        "@oak//cc/ffi:bytes_view",
        "@oak//cc/ffi:error_bindings",
        "@oak//cc/oak_session:config",
        "@oak//cc/oak_session:oak_session_bindings",
        "@oak//proto/session:session_cc_proto",
        "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
    ],
)

_EXCLUSION_LIST = [
    # C++ header files in python tensorflow
    "external/pypi_tensorflow_cpu/site-packages/tensorflow/include/",
    # Duplicate libpython in Bazel solib symlinks directory
    "_solib_k8/",
    # Test directories
    "/tests/",
]

pkg_tar_runfiles(
    name = "runfiles",
    srcs = [":main"],
    exclude_paths = _EXCLUSION_LIST,
    runfiles_prefix = "main.runfiles",
)

pkg_tar(
    name = "tar",
    srcs = [":runfiles"],
)

oci_image(
    name = "oci_image",
    # The python runtime is provided via trusted_py_runtime.
    base = "@distroless_cc_debian12",
    # Set the workdir and entrypoint so that the main binary will be executed
    # from within the runfiles directory. This ensures that the main binary
    # will be able to find its runfiles properly.
    entrypoint = ["./main"],
    tars = [":tar"],
    workdir = "/main.runfiles/__main__/",
)

oci_runtime_bundle(
    name = "oci_runtime_bundle",
    image = ":oci_image",
    visibility = ["//:__pkg__"],
)

pkg_tar_runfiles(
    name = "worker_runfiles",
    srcs = [":worker_main_tensorflow"],
    exclude_paths = _EXCLUSION_LIST,
    runfiles_prefix = "worker_main_tensorflow.runfiles",
)

pkg_tar(
    name = "worker_tar",
    srcs = [":worker_runfiles"],
)

oci_image(
    name = "worker_oci_image",
    base = "@distroless_cc_debian12",
    # Set the workdir and entrypoint so that the main binary will be executed
    # from within the runfiles directory. This ensures that the main binary
    # will be able to find its runfiles properly.
    entrypoint = ["./worker_main_tensorflow"],
    tars = [":worker_tar"],
    workdir = "/worker_main_tensorflow.runfiles/__main__/",
)

oci_runtime_bundle(
    name = "worker_oci_runtime_bundle",
    image = ":worker_oci_image",
    visibility = ["//:__pkg__"],
)

# All release (i.e. production) binaries, along with their names in the
# destination directory.
_RELEASE_BINARIES = {
    # keep-sorted start
    "oci_runtime_bundle.tar": "program_executor_root_tensorflow/container.tar",
    "worker_oci_runtime_bundle.tar": "program_executor_worker_tensorflow/container.tar",
    # keep-sorted end
}

pkg_files(
    name = "release_binaries",
    srcs = _RELEASE_BINARIES.keys(),
    renames = _RELEASE_BINARIES,
)

pkg_install(
    name = "install_release_binaries",
    srcs = [":release_binaries"],
)

# All release and testing binaries, along with their names in the destination
# directory.
_ALL_BINARIES = _RELEASE_BINARIES | {
    # keep-sorted start
    # keep-sorted end
}

pkg_files(
    name = "all_binaries",
    srcs = _ALL_BINARIES.keys(),
    renames = _ALL_BINARIES,
)

pkg_install(
    name = "install_all_binaries",
    srcs = [":all_binaries"],
)
