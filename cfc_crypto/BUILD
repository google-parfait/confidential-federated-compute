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

package(features = ["layering_check"])

rust_library(
    name = "cfc_crypto",
    srcs = glob(["src/*.rs"]),
    visibility = ["//:__subpackages__"],
    deps = [
        "@oak_crates_index//:aes-gcm-siv",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:hpke",
    ],
)

rust_test(
    name = "cfc_crypto_test",
    crate = ":cfc_crypto",
    deps = ["@oak_crates_index//:googletest"],
)
