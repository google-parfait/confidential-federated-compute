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
        integrity = "sha256-jb6NohfkbhR6kCFTzSMu52v9AsCmo3oXOIJHxn2Qvww=",
        strip_prefix = "federated-compute-d9608bf1a16ee03a5451e15c8f3e262617472a38",
        patches = [
            "//third_party/federated_compute:libcppbor.patch",
            "//third_party/federated_compute:visibility.patch",
        ],
        url = "https://github.com/google/federated-compute/archive/d9608bf1a16ee03a5451e15c8f3e262617472a38.tar.gz",
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
        integrity = "sha256-/qfxXbhDGoitrllQ2vQdrYszGrGVVMHIJj1AelkbGn4=",
        strip_prefix = "oak-706193333936def5aace176e12e1f1225bf8db29",
        url = "https://github.com/project-oak/oak/archive/706193333936def5aace176e12e1f1225bf8db29.tar.gz",
    )

cfc_deps = module_extension(implementation = _cfc_deps_impl)
