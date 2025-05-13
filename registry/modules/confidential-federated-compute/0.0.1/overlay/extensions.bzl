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

"""Loads dependencies for Confidential Federated Compute."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _cfc_deps_impl(ctx):  # buildifier: disable=unused-variable
    http_archive(
        name = "federated-compute",
        integrity = "sha256-IdZGjg0WgThV8PrN6/eGy/Gzx/k/kpNBD47rmpqHmL8=",
        strip_prefix = "federated-compute-445ce83a276a4fdc97e27e04c7740fee4abc9d71",
        patches = [
            "//third_party/federated_compute:libcppbor.patch",
            "//third_party/federated_compute:visibility.patch",
        ],
        url = "https://github.com/google/federated-compute/archive/445ce83a276a4fdc97e27e04c7740fee4abc9d71.tar.gz",
    )

    git_repository(
        name = "libcppbor",
        build_file = "@federated-compute//third_party:libcppbor.BUILD.bzl",
        commit = "20d2be8672d24bfb441d075f82cc317d17d601f8",
        patches = ["//third_party/libcppbor:libcppbor.patch"],
        remote = "https://android.googlesource.com/platform/external/libcppbor",
    )

    http_archive(
        name = "oak",
        integrity = "sha256-zJ0Km91oa8jHSyZl2NUneuGxl+SwC/fu429awcxVzGA=",
        strip_prefix = "oak-5332eb16f0a7e29acec62fc22c2b06adbbb4f943",
        url = "https://github.com/project-oak/oak/archive/5332eb16f0a7e29acec62fc22c2b06adbbb4f943.tar.gz",
    )

cfc_deps = module_extension(implementation = _cfc_deps_impl)
