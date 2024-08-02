# Copyright 2024 Google LLC.
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

load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

rust_library(
    name = "sum_service",
    srcs = glob(["src/*.rs"]),
    visibility = ["//examples/sum_enclave_app:__pkg__"],
    deps = [
        "//pipeline_transforms",
        "@crates_index//:byteorder",
        "@oak//micro_rpc",
        "@oak//oak_restricted_kernel_sdk",
    ],
)

rust_test(
    name = "sum_service_test",
    crate = ":sum_service",
    deps = [
        "@crates_index//:anyhow",
        "@crates_index//:coset",
        "@crates_index//:sha2",
        "@oak//oak_attestation",
        "@oak//oak_proto_rust",
    ],
)
