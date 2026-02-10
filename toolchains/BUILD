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

load("@rules_rust//rust:defs.bzl", "rust_library_group")
load("@rules_rust_prost//:defs.bzl", "rust_prost_toolchain")

package(features = ["layering_check"])

rust_library_group(
    name = "prost_runtime",
    deps = ["@oak_crates_index//:prost"],
)

rust_prost_toolchain(
    name = "prost_toolchain_impl",
    prost_plugin = "@crates//:protoc-gen-prost__protoc-gen-prost",
    prost_runtime = ":prost_runtime",
    prost_types = "@oak_crates_index//:prost-types",
    tonic_plugin_flag = "",
)

toolchain(
    name = "prost_toolchain",
    toolchain = "prost_toolchain_impl",
    toolchain_type = "@rules_rust_prost//:toolchain_type",
)
