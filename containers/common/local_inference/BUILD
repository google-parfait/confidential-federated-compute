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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
)

# Placeholder library that carries nothing but this repository's include paths.
#
# TODO: Remove this once containers/local_inference can depend on
# :inference_model directly.
#
# @gemma and @llama_cpp are declared as dev dependencies of this module (see
# MODULE.bazel), so they are invisible to modules that consume this repository,
# and :inference_model therefore cannot be analyzed from those modules. As a
# temporary transitionary measure, containers/local_inference duplicates the
# :inference_model and :mocks rules against these same sources rather than
# forcing all of CFC to take gemma.cpp and llama.cpp as non-dev dependencies.
#
# Bazel only puts a repository on the include path of a target that has a dep
# in that repository, so those duplicated rules would otherwise be unable to
# resolve the "containers/..." includes used by these sources. Depending on
# this empty target supplies that include path.
cc_library(name = "include_path")

cc_library(
    name = "inference_model",
    srcs = ["inference_model.cc"],
    hdrs = ["inference_model.h"],
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
        "@federated_compute//fcp/base:status_converters",
        "@federated_compute//fcp/protos/confidentialcompute:private_inference_cc_proto",
        "@gemma//:gemma_args",
        "@gemma//:gemma_lib",
        "@gemma//:matmul",
        "@gemma//:threading_context",
        "@llama_cpp",
    ],
)

cc_test(
    name = "inference_model_test",
    size = "small",
    srcs = ["inference_model_test.cc"],
    deps = [
        ":inference_model",
        "//containers/common/local_inference/testing:mocks",
        "//testing:matchers",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@gemma//:gemma_lib",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
