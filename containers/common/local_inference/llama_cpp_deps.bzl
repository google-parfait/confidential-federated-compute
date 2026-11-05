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

"""Loads llama_cpp dependency for Confidential Federated Compute."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

visibility("public")

def _llama_cpp_deps_impl(ctx):  # buildifier: disable=unused-variable
    # Pin llama.cpp to release b10400 (matching containers/gcp/MODULE.bazel) so
    # CPU containers support the gemma4 GGUF model architecture.
    # go/keep-sorted start block=yes newline_separated=yes ignore_prefixes=http_archive
    http_archive(
        name = "llama_cpp",
        build_file = "@confidential-federated-compute//containers/common/local_inference:llama_cpp.BUILD.bzl",
        integrity = "sha256-E7X02EvzfwA9Qv6PxvWSvkX0t51U1iDDSRwgUL6xeQw=",
        strip_prefix = "llama.cpp-1f368f354d9edcfea9fd6a1e0989b3e7335a050f",
        url = "https://github.com/ggml-org/llama.cpp/archive/1f368f354d9edcfea9fd6a1e0989b3e7335a050f.tar.gz",
    )
    # go/keep-sorted end

llama_cpp_deps = module_extension(implementation = _llama_cpp_deps_impl)
