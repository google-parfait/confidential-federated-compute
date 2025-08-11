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
load("//:proto_data.bzl", "proto_data")

package(features = ["layering_check"])

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
        "@oak//proto:variant.proto",
        "@oak//proto/attestation:endorsement.proto",
        "@oak//proto/attestation:eventlog.proto",
        "@oak//proto/attestation:evidence.proto",
        "@oak//proto/crypto:certificate.proto",
        "@oak//proto/micro_rpc:options.proto",
    ],
    tools = ["@com_google_protobuf//:protoc"],
    deps = ["@oak//micro_rpc_build"],
)

proto_data(
    name = "insecure_reference_values",
    src = "insecure_reference_values.txtpb",
    out = "insecure_reference_values.binarypb",
    proto_deps = ["@oak//proto/attestation:reference_value_proto"],
    proto_name = "oak.attestation.v1.ReferenceValues",
)

rust_library(
    name = "ledger_service",
    srcs = glob(["src/*.rs"]),
    compile_data = [
        ":insecure_reference_values",
        "//reference_values/ledger:reference_values",
    ],
    crate_features = select({
        "@platforms//os:linux": ["testing"],
        "//conditions:default": [],
    }),
    proc_macro_deps = ["@oak_crates_index//:prost-derive"],
    rustc_env = {
        "REFERENCE_VALUES": "$(execpath //reference_values/ledger:reference_values)",
        "INSECURE_REFERENCE_VALUES": "$(execpath :insecure_reference_values)",
    },
    tags = ["no-clippy"],
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
        "@oak//oak_attestation_verification_types",
        "@oak//oak_crypto",
        "@oak//oak_proto_rust",
        "@oak//oak_restricted_kernel_sdk",
        "@oak//oak_time",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:hashbrown",
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
        "@platforms//os:linux": [
            "@oak//oak_attestation",
            "@oak//oak_sdk/standalone:oak_sdk_standalone",
        ],
        "//conditions:default": [],
    }),
)

rust_test(
    name = "ledger_service_test",
    compile_data = [
        "//ledger_service/testdata:endorsements.binarypb",
        "//ledger_service/testdata:evidence.binarypb",
    ],
    crate = ":ledger_service",
    crate_features = ["testing"],
    # `data` isn't available at compile time, but $(execpath) doesn't work with
    # compile_data. We use both and hope the paths are the same.
    data = [
        "//ledger_service/testdata:endorsements.binarypb",
        "//ledger_service/testdata:evidence.binarypb",
    ],
    rustc_env = {
        "ENDORSEMENTS": "$(execpath //ledger_service/testdata:endorsements.binarypb)",
        "EVIDENCE": "$(execpath //ledger_service/testdata:evidence.binarypb)",
    },
    tags = ["no-clippy"],
    deps = [
        "@oak_crates_index//:googletest",
    ],
)
