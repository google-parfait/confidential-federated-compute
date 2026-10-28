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

load("@pybind11_bazel//:build_defs.bzl", "PYBIND_DEPS", "pybind_library", "pybind_library_test")
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//containers/bazel:embedded_python_env.bzl", "embedded_python_env")
load(":embedded_python_env_test.bzl", "duplicate_distribution_test")

package(licenses = ["notice"])

# pybind_library adds the pybind11 dependency through pybind11_bazel, so this
# module doesn't need to import the @pybind11 repository itself.
pybind_library(
    name = "embedded_python",
    srcs = ["embedded_python.cc"],
    hdrs = ["embedded_python.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@rules_cc//cc/runfiles",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

py_library(
    name = "testdata_greeting",
    testonly = True,
    srcs = [
        "testdata/greeting/__init__.py",
        "testdata/greeting/greeting.py",
    ],
)

# Laid out like a pip package: a top-level package and its .dist-info in a
# site-packages directory that is added to sys.path through `imports`.
py_library(
    name = "testdata_pip_style_package",
    testonly = True,
    srcs = ["testdata/pip_style/site-packages/pip_style_package/__init__.py"],
    data = ["testdata/pip_style/site-packages/pip_style_package-1.0.dist-info/METADATA"],
    imports = ["testdata/pip_style/site-packages"],
)

embedded_python_env(
    name = "test_env",
    testonly = True,
    deps = [
        ":testdata_greeting",
        ":testdata_pip_style_package",
    ],
)

pybind_library_test(
    name = "embedded_python_test",
    srcs = ["embedded_python_test.cc"],
    # LeakSanitizer reports the objects that pybind11 never frees, so tests that
    # initialize an interpreter don't run with ASan.
    tags = ["noasan"],
    deps = [
        ":embedded_python",
        ":test_env",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

# Runs a binary extracted from a pkg_tar, as in a container image: no Bazel
# environment variables, a different working directory, and started through a
# PATH lookup or a symlink.
cc_binary(
    name = "embedded_python_smoke_main",
    testonly = True,
    srcs = ["embedded_python_smoke_main.cc"],
    deps = [
        ":embedded_python",
        ":test_env",
    ] + PYBIND_DEPS,
)

pkg_tar(
    name = "embedded_python_smoke_tar",
    testonly = True,
    srcs = [":embedded_python_smoke_main"],
    include_runfiles = True,
)

py_test(
    name = "embedded_python_smoke_test",
    srcs = ["embedded_python_smoke_test.py"],
    args = ["$(rootpath :embedded_python_smoke_tar)"],
    data = [":embedded_python_smoke_tar"],
    tags = ["noasan"],
)

# The same distribution reachable from two sys.path directories fails the
# build.
py_library(
    name = "testdata_distribution_v1",
    testonly = True,
    data = ["testdata/distribution_v1/site-packages/duplicated_distribution-1.0.dist-info/METADATA"],
    imports = ["testdata/distribution_v1/site-packages"],
)

py_library(
    name = "testdata_distribution_v2",
    testonly = True,
    data = ["testdata/distribution_v2/site-packages/Duplicated.Distribution-2.0.dist-info/METADATA"],
    imports = ["testdata/distribution_v2/site-packages"],
)

embedded_python_env(
    name = "duplicate_distribution_env",
    testonly = True,
    tags = ["manual"],
    deps = [
        ":testdata_distribution_v1",
        ":testdata_distribution_v2",
    ],
)

duplicate_distribution_test(
    name = "duplicate_distribution_test",
    target_under_test = ":duplicate_distribution_env_gen",
)
