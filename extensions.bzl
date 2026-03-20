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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

visibility(["//..."])

def _cfc_dev_deps_impl(ctx):  # buildifier: disable=unused-variable
    # go/keep-sorted start block=yes newline_separated=yes ignore_prefixes=http_archive
    http_archive(
        name = "llama_cpp",
        build_file = "//third_party/llama_cpp:llama_cpp.BUILD.bzl",
        integrity = "sha256-tE6qqF3MiQbSWpTfUDYqTK4KndvvJ0R27udIde7/LE4=",
        patch_args = ["-p1"],
        patches = ["//third_party/llama_cpp:add_build_info.patch"],
        strip_prefix = "llama.cpp-be48528b068111304e4a0bb82c028558b5705f05",
        url = "https://github.com/ggml-org/llama.cpp/archive/be48528b068111304e4a0bb82c028558b5705f05.tar.gz",
    )
    # go/keep-sorted end

cfc_dev_deps = module_extension(implementation = _cfc_dev_deps_impl)
