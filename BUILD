# Copyright 2024 Google LLC.
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
load("@rules_pkg//pkg:install.bzl", "pkg_install")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files")

exports_files([".rustfmt.toml"])

platform(
    name = "remote_platform",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    exec_properties = create_rbe_exec_properties_dict(
        container_image = "docker://gcr.io/cloud-marketplace/google/debian11@sha256:82cf3135ad7f0478870836b8bd303b383c151cb9d6c65512f6dd7b7bc8c70c40",
        os_family = "Linux",
    ),
    parents = ["@local_config_platform//:host"],
)

# All release (i.e. production) binaries, along with their names in the
# destination directory.
_RELEASE_BINARIES = {
    # keep-sorted start
    "//containers/agg_core:oci_runtime_bundle.tar": "agg_core/container.tar",
    "//containers/fed_sql:oci_runtime_bundle.tar": "fed_sql/container.tar",
    "//containers/program_executor_tee:oci_runtime_bundle.tar": "program_executor_tee/container.tar",
    "//containers/program_worker:oci_runtime_bundle.tar": "program_worker/container.tar",
    "//containers/sql_server:oci_runtime_bundle.tar": "sql_server/container.tar",
    "//containers/tff_server:oci_runtime_bundle.tar": "tff_server/container.tar",
    "//ledger_enclave_app": "ledger/binary",
    "//replicated_ledger_enclave_app": "replicated_ledger/binary",
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
    "//containers/confidential_transform_test_concat:oci_runtime_bundle.tar": "confidential_transform_test_concat/container.tar",
    "//containers/kms:oci_runtime_bundle.tar": "kms/container.tar",
    "//containers/test_concat:oci_runtime_bundle.tar": "test_concat/container.tar",
    "//examples/square_enclave_app": "square_example/binary",
    "//examples/sum_enclave_app": "sum_example/binary",
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
