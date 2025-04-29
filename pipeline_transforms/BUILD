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
    name = "pipeline_transforms",
    srcs = glob(["src/**"]),
    crate_features = select({
        "@platforms//os:linux": ["test"],
        "//conditions:default": [],
    }),
    tags = ["no-clippy"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//cfc_crypto",
        "//third_party/federated_compute",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:bitmask",
        "@oak_crates_index//:core2",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:libflate",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:prost-types",
        "@oak_crates_index//:rand",
    ],
)

rust_test(
    name = "pipeline_transforms_test",
    crate = ":pipeline_transforms",
    crate_features = ["test"],
    tags = ["no-clippy"],
    deps = [
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:sha2",
    ],
)
