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

"""Loads dependencies for Confidential Federated Compute."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _cfc_deps_impl(ctx):  # buildifier: disable=unused-variable
    http_archive(
        name = "federated-compute",
        integrity = "sha256-RuobEKidPyUjjwa5WyxnozQej0reSTv7keERgUGCVS8=",
        patches = [
            "//third_party/federated_compute:visibility.patch",
        ],
        strip_prefix = "federated-compute-44a0dfcb498cced1ebd0eb18cba7803deabc2c2e",
        url = "https://github.com/google/federated-compute/archive/44a0dfcb498cced1ebd0eb18cba7803deabc2c2e.tar.gz",
    )

    git_repository(
        name = "libcppbor",
        build_file = "//third_party/libcppbor:BUILD",
        commit = "20d2be8672d24bfb441d075f82cc317d17d601f8",
        patches = ["//third_party/libcppbor:libcppbor.patch"],
        remote = "https://android.googlesource.com/platform/external/libcppbor",
    )

    http_archive(
        name = "oak",
        integrity = "sha256-ZEZ+qDfHWNivUm3z9LkWG5q2F6udbjij8GTgUFmIl/k=",
        strip_prefix = "oak-0beb0de86a546df46365cf4caef3ac8065e304e9",
        url = "https://github.com/project-oak/oak/archive/0beb0de86a546df46365cf4caef3ac8065e304e9.tar.gz",
    )

cfc_deps = module_extension(implementation = _cfc_deps_impl)
