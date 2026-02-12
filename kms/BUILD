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

load("@bazel_toolchains//rules/exec_properties:exec_properties.bzl", "create_rbe_exec_properties_dict")
load("@confidential-federated-compute//:proto_data.bzl", "proto_data")
load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:install.bzl", "pkg_install")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_library")

package(features = ["layering_check"])

exports_files([".rustfmt.toml"])

platform(
    name = "remote_platform",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    exec_properties = create_rbe_exec_properties_dict(
        container_image = "docker://gcr.io/cloud-marketplace/google/debian12@sha256:be15f84c44c4ee68b4f187128f0278df1b0f424c04fb5f08b098967d5b896388",
        os_family = "Linux",
    ),
    parents = ["@local_config_platform//:host"],
)

pkg_files(
    name = "release_binaries",
    srcs = [":oci_runtime_bundle.tar"],
    renames = {":oci_runtime_bundle.tar": "kms/container.tar"},
)

pkg_install(
    name = "install_release_binaries",
    srcs = [":release_binaries"],
)

alias(
    name = "install_all_binaries",
    actual = ":install_release_binaries",
)

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
        "//reference_values",
    ],
    rustc_env = {
        "INSECURE_REFERENCE_VALUES": "$(location :insecure_reference_values)",
        "REFERENCE_VALUES": "$(location //reference_values)",
    },
    rustc_flags = ["-Cpanic=abort"],
    deps = [
        "//key_management_service",
        "//key_management_service:kms_prost_proto",
        "//storage_service:session_v1_service_prost_proto",
        "//storage_service:storage_actor",
        "//storage_service:storage_client",
        "@oak//oak_proto_rust",
        "@oak//oak_sdk/common:oak_sdk_common",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak//oak_time:oak_time_std",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:opentelemetry",
        "@oak_crates_index//:opentelemetry-appender-tracing",
        "@oak_crates_index//:opentelemetry-otlp",
        "@oak_crates_index//:opentelemetry_sdk",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:slog",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
        "@oak_crates_index//:tracing-slog",
        "@oak_crates_index//:tracing-subscriber",
        "@trusted_computations_platform//proto:tcp_proto",
        "@trusted_computations_platform//runtime:tcp_runtime",
    ],
)

oci_image(
    name = "oci_image",
    base = "@distroless_cc_debian12_base",
    entrypoint = ["/main"],
    tars = [":tar"],
)

oci_runtime_bundle(
    name = "oci_runtime_bundle",
    image = ":oci_image",
    visibility = ["//:__pkg__"],
)

rust_library(
    name = "prost_proto_conversion",
    srcs = ["prost_proto_conversion.rs"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//key_management_service:access_policy_prost_proto",
        "//key_management_service:kms_prost_proto",
        "//storage_service:session_v1_service_prost_proto",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:prost",
    ],
)

pkg_tar(
    name = "tar",
    srcs = [":main"],
)
