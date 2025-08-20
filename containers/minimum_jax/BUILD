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
load("@pip//:requirements.bzl", "requirement")
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("//:python_path.bzl", "gen_python_path")

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
    name = "tokens",
    srcs = ["tokens.py"],
    deps = [
        requirement("jax"),
        requirement("sentencepiece"),
    ],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    data = [
        ":tokens",
    ],
    deps = [
        "@pybind11",
        "@rules_python//python/cc:current_py_cc_libs",
    ],
)

pkg_tar(
    name = "main_tar",
    srcs = [":main"],
    include_runfiles = True,
)

pkg_tar(
    name = "checkpoint",
    srcs = [":model.ckpt"],
    package_dir = "/",
)

gen_python_path(
    name = "py_path",
    cc_binary_name = "main",
    py_deps = [":tokens"],
)

oci_image(
    name = "oci_image",
    base = "@python_3_11_slim_bookworm",
    entrypoint = ["/main"],
    env = ":py_path",
    tars = [
        # Install the main cc binary
        ":main_tar",
        ":checkpoint",
    ],
)

oci_load(
    name = "tarball",
    image = ":oci_image",
    repo_tags = [
        "minimum_jax:latest",
    ],
)

oci_runtime_bundle(
    name = "oci_runtime_bundle",
    image = ":oci_image",
)
