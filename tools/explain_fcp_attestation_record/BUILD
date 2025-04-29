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

load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_library", "rust_test")

rust_binary(
    name = "main",
    srcs = ["src/main.rs"],
    deps = [
        ":explain_fcp_attestation_record",
        "//third_party/federated_compute",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:clap",
        "@oak_crates_index//:prost",
    ],
)

rust_library(
    name = "explain_fcp_attestation_record",
    srcs = glob(
        ["src/**"],
        exclude = ["src/main.rs"],
    ),
    tags = ["no-clippy"],
    deps = [
        "//third_party/federated_compute",
        "@oak//oak_attestation_explain",
        "@oak//oak_attestation_verification",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:clap",
        "@oak_crates_index//:prost",
    ],
)

rust_test(
    name = "snapshot_tests",
    srcs = glob(["tests/**/*.rs"]),
    compile_data = glob(["tests/**/*.pb"]),
    crate_root = "tests/snapshot_tests.rs",
    data = glob(["tests/**/*.snap"]) + [":main"],
    env = {
        "EXPLAIN_TOOL": "$(rootpath :main)",
        "INSTA_WORKSPACE_ROOT": ".",
    },
    tags = ["no-clippy"],
    deps = [
        ":explain_fcp_attestation_record",
        "//third_party/federated_compute",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:assert_cmd",
        "@oak_crates_index//:insta",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:tempfile",
    ],
)
