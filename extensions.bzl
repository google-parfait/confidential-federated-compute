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
load("//:stub_tf_proto_library.bzl", "stub_tf_proto_library")

visibility(["//..."])

def _cfc_deps_impl(ctx):  # buildifier: disable=unused-variable
    # go/keep-sorted start block=yes newline_separated=yes ignore_prefixes=git_repository,http_archive,stub_tf_proto_library
    http_archive(
        name = "bazel_toolchains",
        sha256 = "294cdd859e57fcaf101d4301978c408c88683fbc46fbc1a3829da92afbea55fb",
        strip_prefix = "bazel-toolchains-8c717f8258cd5f6c7a45b97d974292755852b658",
        url = "https://github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
    )

    http_archive(
        name = "federated-compute",
        integrity = "sha256-AgQnf/4BEpigoN7H3yuTW5aAZnYPLKz4CU9X8yRKRR0=",
        patches = ["//third_party/federated_compute:visibility.patch"],
        strip_prefix = "federated-compute-fa94d08b3f1203025b889b2579f2232b7fe99803",
        url = "https://github.com/google/federated-compute/archive/fa94d08b3f1203025b889b2579f2232b7fe99803.tar.gz",
    )

    git_repository(
        name = "libcppbor",
        build_file = "//third_party/libcppbor:libcppbor.BUILD",
        commit = "20d2be8672d24bfb441d075f82cc317d17d601f8",
        patches = ["//third_party/libcppbor:libcppbor.patch"],
        remote = "https://android.googlesource.com/platform/external/libcppbor",
    )

    stub_tf_proto_library(
        name = "org_tensorflow",
    )

    http_archive(
        name = "org_tensorflow_federated",
        integrity = "sha256-LRtgjagA16SepVffwpMCh8zfT60rzGSYRekKAGaQ2ss=",
        patches = ["//third_party/tensorflow_federated:proto_library.patch"],
        strip_prefix = "tensorflow-federated-590104cb7c358ee5c4efa8937e0ccf93a5925265",
        url = "https://github.com/google-parfait/tensorflow-federated/archive/590104cb7c358ee5c4efa8937e0ccf93a5925265.tar.gz",
    )
    # go/keep-sorted end

cfc_deps = module_extension(implementation = _cfc_deps_impl)

def _cfc_dev_deps_impl(ctx):  # buildifier: disable=unused-variable
    # go/keep-sorted start block=yes newline_separated=yes ignore_prefixes=http_archive
    http_archive(
        name = "com_google_cc_differential_privacy",
        sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
        strip_prefix = "differential-privacy-3.0.0/cc",
        url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
    )

    http_archive(
        name = "com_google_differential_privacy",
        sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
        strip_prefix = "differential-privacy-3.0.0",
        url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
    )

    http_archive(
        name = "llama_cpp",
        build_file = "//third_party/llama_cpp:llama_cpp.BUILD.bzl",
        integrity = "sha256-tE6qqF3MiQbSWpTfUDYqTK4KndvvJ0R27udIde7/LE4=",
        patch_args = ["-p1"],
        patches = ["//third_party/llama_cpp:add_build_info.patch"],
        strip_prefix = "llama.cpp-be48528b068111304e4a0bb82c028558b5705f05",
        url = "https://github.com/ggml-org/llama.cpp/archive/be48528b068111304e4a0bb82c028558b5705f05.tar.gz",
    )

    http_archive(
        name = "sysroot",
        sha256 = "f58c289b3ccb28895ad8ca408ac366e709037088e8b5c28aca18212adc18c31e",
        url = "https://github.com/google-parfait/confidential-federated-compute/releases/download/sysroot-20250618/sysroot.tar.xz",
    )
    # go/keep-sorted end

cfc_dev_deps = module_extension(implementation = _cfc_dev_deps_impl)
