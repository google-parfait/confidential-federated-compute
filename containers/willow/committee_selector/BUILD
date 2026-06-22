# Copyright 2026 The Trusted Computations Platform Authors.
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

load("@confidential-federated-compute//:proto_data.bzl", "proto_data")
load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_rust//cargo:defs.bzl", "cargo_build_script")
load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_library", "rust_test")

proto_data(
    name = "insecure_reference_values",
    src = "insecure_reference_values.txtpb",
    out = "insecure_reference_values.binarypb",
    proto_deps = ["@oak//proto/attestation:reference_value_proto"],
    proto_name = "oak.attestation.v1.ReferenceValues",
)

rust_binary(
    name = "main",
    srcs = ["main.rs"],
    compile_data = [
        ":insecure_reference_values",
        "//committee_selector/reference_values",
    ],
    rustc_env = {
        "INSECURE_REFERENCE_VALUES": "$(location :insecure_reference_values)",
        "REFERENCE_VALUES": "$(location //committee_selector/reference_values)",
    },
    deps = [
        ":willow_committee_selector_service",
        "@oak//oak_proto_rust",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:log",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tonic",
        "@secure_aggregation//willow/protocol:willow_v1_decryptor",
        "@trusted_computations_platform//proto:tcp_proto",
        "@trusted_computations_platform//runtime:tcp_runtime",
    ],
)

pkg_tar(
    name = "tar",
    srcs = [":main"],
    package_dir = "/usr/local/bin",
)

oci_image(
    name = "image",
    base = "@distroless_cc_debian12_base",
    entrypoint = ["/usr/local/bin/main"],
    tars = [":tar"],
)

oci_runtime_bundle(
    name = "bundle",
    image = ":image",
    visibility = ["//:__pkg__"],
)

cargo_build_script(
    name = "build",
    srcs = ["build.rs"],
    build_script_env = {
        "PROTOC": "$(execpath @com_google_protobuf//:protoc)",
    },
    data = glob(["proto/**"]) + [
        "@com_google_protobuf//:descriptor_proto_srcs",
        "@com_google_protobuf//:well_known_type_protos",
    ],
    tools = ["@com_google_protobuf//:protoc"],
    deps = ["@oak//micro_rpc_build"],
)

rust_library(
    name = "willow_committee_selector_service",
    srcs = glob(["src/*.rs"]),
    crate_features = select({
        "@platforms//os:linux": [
            "testing",
            "std",
        ],
        "//conditions:default": [],
    }),
    proc_macro_deps = ["@oak_crates_index//:prost-derive"],
    visibility = [
        "//containers/willow/committee_selector:__subpackages__",
    ],
    deps = [
        ":build",
        "@com_google_protobuf//rust:protobuf",
        "@oak//micro_rpc",
        "@oak//oak_attestation_verification:oak_attestation_verification_with_regex",
        "@oak//oak_proto_rust",
        "@oak//oak_restricted_kernel_sdk",
        "@oak_crates_index//:hashbrown",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:prost-types",
        "@oak_crates_index//:rand",
        "@oak_crates_index//:sha2",
        "@oak_crates_index//:slog",
        "@secure_aggregation//willow/api:aggregation_config",
        "@secure_aggregation//willow/api:proto_serialization_traits",
        "@secure_aggregation//willow/crypto:ahe_traits",
        "@secure_aggregation//willow/crypto:kahe_traits",
        "@secure_aggregation//willow/crypto:shell_ahe",
        "@secure_aggregation//willow/crypto:shell_kahe",
        "@secure_aggregation//willow/crypto:shell_parameters",
        "@secure_aggregation//willow/crypto:shell_vahe",
        "@secure_aggregation//willow/crypto:vahe_traits",
        "@secure_aggregation//willow/proto/willow:key_rust_proto",
        "@secure_aggregation//willow/proto/willow:messages_rust_proto",
        "@secure_aggregation//willow/protocol:decryptor_traits",
        "@secure_aggregation//willow/protocol:messages",
        "@secure_aggregation//willow/protocol:willow_v1_decryptor",
        "@trusted_computations_platform//proto:tcp_proto",
        "@trusted_computations_platform//runtime:tcp_runtime",
    ] + select({
        "@platforms//os:linux": ["@oak_crates_index//:slog-term"],
        "//conditions:default": [],
    }),
)

rust_test(
    name = "willow_committee_selector_integration_test",
    srcs = ["tests/integration.rs"],
    crate_features = [
        "testing",
        "std",
    ],
    env = {
        "RUST_TEST_THREADS": "1",
    },
    deps = [
        ":willow_committee_selector_service",
        "@com_google_protobuf//rust:protobuf",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:mockall",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:slog",
        "@secure_aggregation//willow/api:aggregation_config",
        "@secure_aggregation//willow/api:proto_serialization_traits",
        "@secure_aggregation//willow/crypto:ahe_traits",
        "@secure_aggregation//willow/crypto:kahe_traits",
        "@secure_aggregation//willow/crypto:shell_kahe",
        "@secure_aggregation//willow/crypto:shell_parameters",
        "@secure_aggregation//willow/crypto:shell_vahe",
        "@secure_aggregation//willow/proto/shell:shell_ciphertexts_rust_proto",
        "@secure_aggregation//willow/proto/willow:key_rust_proto",
        "@secure_aggregation//willow/proto/willow:messages_rust_proto",
        "@secure_aggregation//willow/protocol:client_traits",
        "@secure_aggregation//willow/protocol:messages",
        "@secure_aggregation//willow/protocol:server_traits",
        "@secure_aggregation//willow/protocol:verifier_traits",
        "@secure_aggregation//willow/protocol:willow_v1_client",
        "@secure_aggregation//willow/protocol:willow_v1_decryptor",
        "@secure_aggregation//willow/protocol:willow_v1_server",
        "@secure_aggregation//willow/protocol:willow_v1_verifier",
        "@trusted_computations_platform//integration:tcp_integration",
        "@trusted_computations_platform//proto:tcp_proto",
    ],
)

rust_test(
    name = "willow_committee_selector_selector_test",
    srcs = ["tests/selector.rs"],
    crate_features = [
        "testing",
        "std",
    ],
    env = {
        "RUST_TEST_THREADS": "1",
    },
    deps = [
        ":willow_committee_selector_service",
        "@oak_crates_index//:slog",
    ],
)
