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

load("@rules_rust//cargo:defs.bzl", "cargo_build_script")
load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")

cargo_build_script(
    name = "build",
    srcs = ["build.rs"],
    build_script_env = {
        "PROTOC": "$(execpath @com_google_protobuf//:protoc)",
        "DESCRIPTOR_PROTO": "$(location @com_google_protobuf//:descriptor_proto_srcs)",
        "EVIDENCE_PROTO": "$(location @oak//proto/attestation:evidence.proto)",
        "LEDGER_PROTO": "$(location @federated-compute//fcp/protos/confidentialcompute:ledger.proto)",
    },
    data = glob(["proto/**"]) + [
        "@com_google_protobuf//:descriptor_proto_srcs",
        "@com_google_protobuf//:well_known_type_protos",
        "@federated-compute//fcp/protos/confidentialcompute:ledger.proto",
        "@oak//proto:digest.proto",
        "@oak//proto/attestation:endorsement.proto",
        "@oak//proto/attestation:eventlog.proto",
        "@oak//proto/attestation:evidence.proto",
        "@oak//proto/micro_rpc:options.proto",
    ],
    tools = ["@com_google_protobuf//:protoc"],
    deps = [
        "@oak//micro_rpc_build",
        "@oak//oak_proto_build_utils",
    ],
)

rust_library(
    name = "ledger_service",
    srcs = glob(["src/*.rs"]),
    crate_features = select({
        "@platforms//os:linux": ["testing"],
        "//conditions:default": [],
    }),
    proc_macro_deps = ["@oak_crates_index//:prost-derive"],
    visibility = [
        "//ledger_enclave_app:__pkg__",
        "//replicated_ledger_enclave_app:__pkg__",
    ],
    deps = [
        ":build",
        "//cfc_crypto",
        "//third_party/federated_compute",
        "@oak//micro_rpc",
        "@oak//oak_attestation_verification",
        "@oak//oak_proto_rust",
        "@oak//oak_restricted_kernel_sdk",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:hpke",
        "@oak_crates_index//:p256",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:prost-types",
        "@oak_crates_index//:rand",
        "@oak_crates_index//:rangemap",
        "@oak_crates_index//:sha2",
        "@oak_crates_index//:slog",
        "@trusted_computations_platform//runtime:tcp_runtime",
    ] + select({
        "@platforms//os:linux": ["@oak//oak_attestation"],
        "//conditions:default": [],
    }),
)

rust_test(
    name = "ledger_service_test",
    crate = ":ledger_service",
    crate_features = ["testing"],
    deps = [
        "@oak_crates_index//:googletest",
    ],
)
