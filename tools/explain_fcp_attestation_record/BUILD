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
load("@rules_rust_prost//:defs.bzl", "rust_prost_library")

package(features = ["layering_check"])

rust_binary(
    name = "main",
    srcs = ["src/main.rs"],
    deps = [
        ":explain_fcp_attestation_record",
        ":verification_record_prost_proto",
        "@crates//:clap",
        "@crates//:reqwest",
        "@crates//:tokio",
        "@oak_crates_index//:anyhow",
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
        ":fed_sql_container_config_prost_proto",
        ":messages_prost_proto",
        ":signed_endorsements_prost_proto",
        ":verification_record_prost_proto",
        "@crates//:clap",
        "@crates//:futures-util",
        "@crates//:integer-encoding",
        "@crates//:reqwest",
        "@crates//:tokio",
        "@oak//oak_attestation_explain",
        "@oak//oak_attestation_verification",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:hex",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:prost-types",
        "@oak_crates_index//:sha2",
    ],
)

rust_test(
    name = "snapshot_tests",
    srcs = glob(["tests/**/*.rs"]),
    compile_data = glob([
        "tests/**/*.pb",
        "tests/**/*.pem",
    ]),
    crate_root = "tests/snapshot_tests.rs",
    data = glob(["tests/**/*.snap"]) + [":main"],
    env = {
        "EXPLAIN_TOOL": "$(rootpath :main)",
        "INSTA_WORKSPACE_ROOT": ".",
    },
    tags = ["no-clippy"],
    deps = [
        ":explain_fcp_attestation_record",
        ":messages_prost_proto",
        ":signed_endorsements_prost_proto",
        ":verification_record_prost_proto",
        "@crates//:assert_cmd",
        "@crates//:axum",
        "@crates//:axum-server",
        "@crates//:insta",
        "@crates//:reqwest",
        "@crates//:rustls",
        "@crates//:tempfile",
        "@crates//:tokio",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:hex",
        "@oak_crates_index//:mockall",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:sha2",
    ],
)

rust_prost_library(
    name = "verification_record_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:verification_record_proto",
)

rust_prost_library(
    name = "fed_sql_container_config_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:fed_sql_container_config_proto",
)

rust_prost_library(
    name = "messages_prost_proto",
    proto = "@oak//proto/session:messages_proto",
)

rust_prost_library(
    name = "signed_endorsements_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:signed_endorsements_proto",
)
