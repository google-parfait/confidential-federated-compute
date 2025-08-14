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

load("@aspect_rules_py//py:defs.bzl", "py_binary", "py_image_layer")
load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@pip//:requirements.bzl", "requirement")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

# Define a set of PIP-provided deps to use.
compile_pip_requirements(
    name = "requirements",
    timeout = "moderate",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
        "--extra-index-url=https://download.pytorch.org/whl/cu128",
    ],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    data = [
        ":model.ckpt",
        ":tokenizer.model",
    ],
    main = "main.py",
    deps = [
        requirement("absl_py"),
        requirement("numpy"),
        requirement("sentencepiece"),
        requirement("torch"),
        "@gemma_pytorch//:gemma",
    ],
)

py_image_layer(
    name = "main_py",
    binary = ":main",
)

# model.ckpt and tokenizer.model files are downloaded from kaggle.
# These files are too big to submit to github.
pkg_tar(
    name = "checkpoints",
    srcs = [
        ":model.ckpt",
        ":tokenizer.model",
    ],
    package_dir = "/",
)

oci_image(
    name = "oci_image",
    base = "@distroless_cc_debian12_base",
    entrypoint = ["/main"],
    tars = [
        # Install the APT deps.
        "@bookworm//:flat",
        # Install the main python binary
        ":main_py",
        ":checkpoints",
    ],
)

oci_load(
    name = "tarball",
    image = ":oci_image",
    repo_tags = [
        "minimum_pytorch:latest",
    ],
)

oci_runtime_bundle(
    name = "oci_runtime_bundle",
    image = ":oci_image",
)
