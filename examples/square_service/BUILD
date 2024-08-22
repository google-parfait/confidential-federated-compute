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
    name = "square_service",
    srcs = glob(["src/*.rs"]),
    visibility = ["//examples/square_enclave_app:__pkg__"],
    deps = [
        "//pipeline_transforms",
        "//third_party/federated_compute",
        "@oak//micro_rpc",
        "@oak//oak_restricted_kernel_sdk",
        "@oak_crates_index//:byteorder",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:prost-types",
    ],
)

rust_test(
    name = "square_service_test",
    crate = ":square_service",
    deps = [
        "//cfc_crypto",
        "@oak//oak_attestation",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:sha2",
    ],
)
