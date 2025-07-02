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

load("@grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_proto_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_proto//proto:defs.bzl", "proto_library")

proto_library(
    name = "test_cuda_proto",
    srcs = ["test_cuda.proto"],
    deps = [],
)

cc_proto_library(
    name = "test_cuda_cc_proto",
    deps = [":test_cuda_proto"],
)

cc_grpc_library(
    name = "test_cuda_cc_grpc",
    srcs = [":test_cuda_proto"],
    grpc_only = True,
    deps = [":test_cuda_cc_proto"],
)

cuda_library(
    name = "kernel",
    srcs = ["kernel.cu"],
    hdrs = ["kernel.h"],
    copts = [
        "-Wno-error",
        "-stdlib=libstdc++",
    ],
    # curand needs to be manually added for now.
    # It's a known issue for rules_cuda, https://github.com/bazel-contrib/rules_cuda/issues/337
    deps = ["@cuda_curand//:curand"],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    linkstatic = True,
    deps = [
        ":kernel",
        ":test_cuda_cc_grpc",
        ":test_cuda_cc_proto",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@grpc//:grpc++",
        "@oak//cc/containers/sdk:orchestrator_client",
    ],
)

pkg_tar(
    name = "main_tar",
    srcs = [":main"],
    include_runfiles = True,
)

oci_image(
    name = "oci_image",
    base = "@distroless_cc_debian12_base",
    entrypoint = ["/main"],
    tars = [
        # Install the main cc binary
        ":main_tar",
    ],
)

# bazel run :tarball to load the image
oci_load(
    name = "tarball",
    image = ":oci_image",
    repo_tags = [
        "minimum_rules_cuda:latest",
    ],
)

oci_runtime_bundle(
    name = "oci_runtime_bundle",
    image = ":oci_image",
)
