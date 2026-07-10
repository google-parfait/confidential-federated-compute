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

cargo_build_script(
    name = "build",
    srcs = ["build.rs"],
    build_script_env = {
        "PROTOC": "$(execpath @com_google_protobuf//:protoc)",
        "COMMITTEE_SELECTOR_PROTO_FILE": "$(location //committee_selector:proto/willow_committee_selector.proto)",
        "REPUTABLE_DECRYPTOR_PROTO_FILE": "$(location //reputable_decryptor:proto/willow_reputable_decryptor.proto)",
        "SECURE_AGGREGATION_PROTO_PATH": "$(location @secure_aggregation//willow/proto/shell:ciphertexts.proto)",
        "SHELL_ENCRYPTION_PROTO_PATH": "$(location @shell-encryption//shell_encryption/rns:rns_serialization.proto)",
    },
    data = [
        "proto/multi_decryptor.proto",
        "//committee_selector:proto/willow_committee_selector.proto",
        "//reputable_decryptor:proto/willow_reputable_decryptor.proto",
        "@com_google_protobuf//:descriptor_proto_srcs",
        "@com_google_protobuf//:well_known_type_protos",
        "@secure_aggregation//willow/proto/shell:ciphertexts.proto",
        "@secure_aggregation//willow/proto/shell:parameters.proto",
        "@secure_aggregation//willow/proto/willow:aggregation_config.proto",
        "@secure_aggregation//willow/proto/willow:messages.proto",
        "@secure_aggregation//willow/proto/zk:proofs.proto",
        "@shell-encryption//shell_encryption:serialization.proto",
        "@shell-encryption//shell_encryption/rns:rns_serialization.proto",
    ],
    tools = ["@com_google_protobuf//:protoc"],
    deps = [
        "@oak//micro_rpc_build",
    ],
)

rust_library(
    name = "willow_multi_decryptor",
    srcs = ["src/actor.rs"],
    deps = [
        ":build",
        "//committee_selector:willow_committee_selector_service",
        "//reputable_decryptor:willow_reputable_decryptor_service",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:slog",
        "@trusted_computations_platform//proto:tcp_proto",
        "@trusted_computations_platform//runtime:tcp_runtime",
    ],
)

rust_binary(
    name = "main",
    srcs = ["main.rs"],
    compile_data = [
        ":insecure_reference_values",
        "//multi_decryptor/reference_values",
    ],
    rustc_env = {
        "INSECURE_REFERENCE_VALUES": "$(location :insecure_reference_values)",
        "REFERENCE_VALUES": "$(location //multi_decryptor/reference_values)",
    },
    deps = [
        ":willow_multi_decryptor",
        "@oak//oak_proto_rust",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:log",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tonic",
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

rust_test(
    name = "willow_multi_decryptor_integration_test",
    srcs = ["tests/integration.rs"],
    crate_features = [
        "testing",
        "std",
    ],
    env = {
        "RUST_TEST_THREADS": "1",
    },
    deps = [
        ":willow_multi_decryptor",
        "//committee_selector:willow_committee_selector_service",
        "//reputable_decryptor:willow_reputable_decryptor_service",
        "@com_google_protobuf//rust:protobuf",
        "@oak_crates_index//:googletest",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:slog",
        "@trusted_computations_platform//integration:tcp_integration",
        "@trusted_computations_platform//proto:tcp_proto",
    ],
)
