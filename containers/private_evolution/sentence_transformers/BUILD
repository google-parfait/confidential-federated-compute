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

load("@pip//:requirements.bzl", "requirement")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

# Define a set of PIP-provided deps to use.
compile_pip_requirements(
    name = "requirements",
    timeout = "moderate",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)

py_library(
    name = "generate_embedding",
    srcs = ["generate_embedding.py"],
    deps = [
        requirement("sentence-transformers"),
    ],
)

cc_library(
    name = "model_delegate",
    hdrs = ["model_delegate.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "py_model_delegate",
    srcs = ["py_model_delegate.cc"],
    hdrs = ["py_model_delegate.h"],
    data = [":generate_embedding"],
    deps = [
        ":model_delegate",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@pybind11",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

cc_library(
    name = "archive_utils",
    srcs = ["archive_utils.cc"],
    hdrs = ["archive_utils.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@libarchive//libarchive:archive",
    ],
)

cc_test(
    name = "archive_utils_test",
    srcs = ["archive_utils_test.cc"],
    data = ["//test:archive_utils_test_resource"],
    deps = [
        ":archive_utils",
        "@bazel_tools//tools/cpp/runfiles",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
