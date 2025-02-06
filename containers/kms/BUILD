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

load("@rules_rust//proto/prost:defs.bzl", "rust_prost_library")
load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

rust_library(
    name = "matchers",
    testonly = True,
    srcs = ["matchers.rs"],
    deps = [
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:tonic",
    ],
)

rust_test(
    name = "matchers_test",
    crate = ":matchers",
)

rust_library(
    name = "storage",
    srcs = ["storage.rs"],
    deps = [
        ":storage_prost_proto",
        "@oak//oak_attestation_verification_types",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:log",
        "@oak_crates_index//:tonic",
    ],
)

rust_test(
    name = "storage_test",
    srcs = ["storage_test.rs"],
    env = {"RUST_LOG": "debug"},
    deps = [
        ":matchers",
        ":storage",
        ":storage_prost_proto",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:test-log",
        "@oak_crates_index//:tonic",
    ],
)

# buildifier: disable=native-proto
proto_library(
    name = "storage_proto",
    srcs = ["storage.proto"],
    data = [":storage_transform"],
    deps = [
        "@com_google_protobuf//:timestamp_proto",
    ],
)

rust_prost_library(
    name = "storage_prost_proto",
    proto = ":storage_proto",
)
