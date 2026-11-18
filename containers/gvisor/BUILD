# Copyright 2026 Google LLC.
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

"""Experimental container: runs a customer rootfs layer inside gVisor, once per
input. See README.md."""

load("@bazel_toolchains//rules/exec_properties:exec_properties.bzl", "create_rbe_exec_properties_dict")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:install.bzl", "pkg_install")
load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_files", "strip_prefix")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

# Referenced by the repo-wide .bazelrc.common (same as containers/mauve_score).
platform(
    name = "remote_platform",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    exec_properties = create_rbe_exec_properties_dict(
        container_image = "docker://gcr.io/cloud-marketplace/google/debian12@sha256:be15f84c44c4ee68b4f187128f0278df1b0f424c04fb5f08b098967d5b896388",
        os_family = "Linux",
    ),
    parents = ["@local_config_platform//:host"],
)

cc_library(
    name = "process_runner",
    srcs = ["process_runner.cc"],
    hdrs = ["process_runner.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@reproc",
    ],
)

cc_test(
    name = "process_runner_test",
    srcs = ["process_runner_test.cc"],
    deps = [
        ":process_runner",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/time",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

proto_library(
    name = "gvisor_report_proto",
    srcs = ["gvisor_report.proto"],
)

cc_proto_library(
    name = "gvisor_report_cc_proto",
    deps = [":gvisor_report_proto"],
)

cc_library(
    name = "gvisor_do_fn",
    srcs = ["gvisor_do_fn.cc"],
    hdrs = ["gvisor_do_fn.h"],
    deps = [
        ":gvisor_report_cc_proto",
        ":process_runner",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:any_cc_proto",
        "@com_google_protobuf//:protobuf",
        "@confidential-federated-compute//containers:session",
        "@confidential-federated-compute//containers/fns",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":gvisor_do_fn",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@confidential-federated-compute//containers/fns",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

pkg_tar(
    name = "main_tar",
    srcs = [":main"],
    include_runfiles = True,
)

# runsc plus its gvisor-bin/ sidecars, installed under /usr/local/gvisor.
pkg_files(
    name = "gvisor_files",
    srcs = ["@gvisor_release//:all_files"],
    attributes = pkg_attributes(mode = "0755"),
    prefix = "usr/local/gvisor",
    strip_prefix = strip_prefix.from_root(),
)

pkg_tar(
    name = "gvisor_tar",
    srcs = [":gvisor_files"],
)

oci_image(
    name = "oci_image",
    base = "@distroless_cc_debian12_base",
    entrypoint = ["/main"],
    tars = [
        # bsdtar and coreutils, which the distroless base image doesn't have.
        "@bookworm//:flat",
        ":gvisor_tar",
        ":main_tar",
    ],
)

# Oak runs the trusted container with rootless runc, so these capabilities are
# scoped to the container's own user namespace (which maps a single uid). On
# top of Oak's default capabilities (CAP_AUDIT_WRITE, CAP_KILL and
# CAP_NET_BIND_SERVICE, the same set runc uses by default), runsc needs exactly
# two (found by dropping capabilities one at a time):
# - CAP_SYS_ADMIN: runsc creates the gofer/sandbox namespaces and the minimal
#   chroot.
# - CAP_SETFCAP: runsc maps uid 0 into the nested user namespace it creates for
#   the gofer and the sandbox. Linux >= 5.12 requires it in the parent user
#   namespace (kernel/user_namespace.c, verify_root_map()).
_CAPS = '["CAP_SYS_ADMIN", "CAP_SETFCAP"]'

_CONFIG_PATCH = " | ".join([
    ".process.capabilities.bounding = ((.process.capabilities.bounding // []) + {caps} | unique)".format(caps = _CAPS),
    ".process.capabilities.effective = ((.process.capabilities.effective // []) + {caps} | unique)".format(caps = _CAPS),
    ".process.capabilities.permitted = ((.process.capabilities.permitted // []) + {caps} | unique)".format(caps = _CAPS),
])

# TODO: Switch to Oak's app_bundle. It replaces the deprecated
# oci_runtime_bundle, but its config.json template is fixed, so it can't add
# the capabilities above yet.
oci_runtime_bundle(
    name = "oci_runtime_bundle",
    config_patch = _CONFIG_PATCH,
    image = ":oci_image",
    visibility = ["//:__pkg__"],
)

# All release binaries with their destination paths.
_RELEASE_BINARIES = {
    # keep-sorted start
    "//:oci_runtime_bundle.tar": "gvisor/container.tar",
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
