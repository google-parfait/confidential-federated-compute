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

load("@bazel_toolchains//rules/exec_properties:exec_properties.bzl", "create_rbe_exec_properties_dict")
load("@oak//oak_containers/app_base:defs.bzl", "app_bundle")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_pkg//pkg:install.bzl", "pkg_install")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files")

package(default_visibility = ["//visibility:public"])

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

# The :inference_model and :mocks rules below duplicate the rules of the same
# name in @confidential-federated-compute//containers/common/local_inference,
# compiling the very same sources.
#
# TODO: Depend on those rules directly instead of duplicating them.
#
# This is a temporary transitionary measure: @gemma and @llama_cpp are dev
# dependencies of the confidential-federated-compute module, so they are
# invisible to this module and the rules over there cannot be analyzed from
# here. Duplicating two rules is cheaper for now than forcing all of CFC to
# take gemma.cpp and llama.cpp as non-dev dependencies.
#
# The :include_path dep is what gives these rebuilt rules the include path of
# the repository the sources come from; see its definition for details.
cc_library(
    name = "inference_model",
    srcs = ["@confidential-federated-compute//containers/common/local_inference:inference_model.cc"],
    hdrs = ["@confidential-federated-compute//containers/common/local_inference:inference_model.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@confidential-federated-compute//containers/common/local_inference:include_path",
        "@federated_compute//fcp/base:status_converters",
        "@federated_compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@gemma//:gemma_args",
        "@gemma//:gemma_lib",
        "@gemma//:matmul",
        "@gemma//:threading_context",
        "@llama_cpp",
    ],
)

cc_library(
    name = "local_batched_inference_engine",
    srcs = ["local_batched_inference_engine.cc"],
    hdrs = ["local_batched_inference_engine.h"],
    deps = [
        ":inference_model",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@confidential-federated-compute//containers/common/inference:batched_inference_engine",
        "@confidential-federated-compute//containers/fns",
        "@federated_compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
    ],
)

cc_library(
    name = "mocks",
    testonly = True,
    hdrs = ["@confidential-federated-compute//containers/common/local_inference/testing:mocks.h"],
    deps = [
        ":inference_model",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "local_batched_inference_engine_test",
    srcs = ["local_batched_inference_engine_test.cc"],
    deps = [
        ":inference_model",
        ":local_batched_inference_engine",
        ":mocks",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:any_cc_proto",
        "@confidential-federated-compute//containers:session",
        "@confidential-federated-compute//containers/common/inference:batched_inference_fn",
        "@confidential-federated-compute//containers/common/inference:batched_inference_test_utils",
        "@confidential-federated-compute//containers/fns",
        "@confidential-federated-compute//containers/testing",
        "@confidential-federated-compute//testing:parse_text_proto",
        "@federated_compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":local_batched_inference_engine",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@confidential-federated-compute//containers/common/inference:batched_inference_engine",
        "@confidential-federated-compute//containers/common/inference:batched_inference_server",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
)

app_bundle(
    name = "app_bundle",
    binary = ":main",
)

# All release (i.e. production) binaries, along with their names in the
# destination directory. Used by scripts/build.sh for Kokoro release builds.
_RELEASE_BINARIES = {
    # keep-sorted start
    ":app_bundle": "batched_inference_local/container.tar",
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
