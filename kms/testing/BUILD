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

load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

package(features = ["layering_check"])

rust_library(
    name = "matchers",
    testonly = True,
    srcs = ["matchers.rs"],
    visibility = ["//:__subpackages__"],
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
    name = "session_test_utils",
    testonly = True,
    srcs = ["session_test_utils.rs"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//:prost_proto_conversion",
        "//key_management_service:access_policy_prost_proto",
        "//key_management_service:kms_prost_proto",
        "@oak//oak_attestation_types",
        "@oak//oak_sdk/common:oak_sdk_common",
        "@oak//oak_sdk/standalone:oak_sdk_standalone",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:p256",
        "@oak_crates_index//:rand_core",
    ],
)
