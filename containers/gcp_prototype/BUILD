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

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_rust//rust:defs.bzl", "rust_library")
load(":build_defs.bzl", "define_load_runner", "generate_policy", "image_digest_flag")

proto_library(
    name = "policy_proto",
    srcs = ["policy.proto"],
)

cc_proto_library(
    name = "policy_cc_proto",
    deps = [":policy_proto"],
)

# C++ library for HTTP client utilities
cc_library(
    name = "http_client",
    srcs = ["http_client.cc"],
    hdrs = ["http_client.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@curl",  # Depends on curl
    ],
)

# Attestation providers
cc_library(
    name = "attestation_cc",
    srcs = ["attestation.cc"],
    hdrs = ["attestation.h"],
    deps = [
        ":http_client",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

# Shared session utilities (message pumping, handshake)
cc_library(
    name = "session_utils",
    srcs = ["session_utils.cc"],
    hdrs = ["session_utils.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@oak//cc/oak_session:client_session",
        "@oak//cc/oak_session:server_session",
        "@oak//proto/services:session_v1_service_cc_grpc",
    ],
)

# Rust library for the server session config
rust_library(
    name = "server_session_config_rs",
    srcs = ["server_session_config.rs"],
    deps = [
        "@oak//oak_attestation_types",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:ecdsa",
        "@oak_crates_index//:p256",
    ],
)

# C++ library that acts as a bridge to the Rust code.
cc_library(
    name = "server_session_config_cc",
    hdrs = ["server_session_config.h"],
    deps = [":server_session_config_rs"],
)

# Llama.cpp integration factored into a helper library
cc_library(
    name = "inference_engine",
    srcs = ["inference_engine.cc"],
    hdrs = ["inference_engine.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llama_cpp",
    ],
)

# The gRPC server that will run in the container
cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":inference_engine",
        ":server_session_config_cc",
        ":attestation_cc",
        ":session_utils",  # Added dependency
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_github_grpc_grpc//:grpc++",
        "@oak//cc/oak_session:server_session",
        "@oak//proto/services:session_v1_service_cc_grpc",
        # Dependencies needed for SignatureConfig::Register()
        "@com_google_tink//tink/signature:signature_config",
        "@com_google_tink//tink/config:tink_config",
        "@oak//cc/ffi:rust_bytes",  # Needed for rust::Vec used by ReadToRustBytes
    ],
)

# C++ library for JSON utilities
cc_library(
    name = "json_util",
    srcs = ["json_util.cc"],
    hdrs = ["json_util.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",  # Depends on nlohmann::json
    ],
)

# C++ library for the client-side verifier implementation
cc_library(
    name = "verifier_cc",
    srcs = ["verifier.cc"],
    hdrs = ["verifier.h"],
    deps = [
        ":policy_cc_proto",
        ":http_client",
        ":json_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",  # For absl::Minutes

        # Oak Verifier interface and protos
        "@oak//cc/attestation/verification:attestation_verifier",
        "@oak//proto/attestation:evidence_cc_proto",
        "@oak//proto/attestation:endorsement_cc_proto",
        "@oak//proto/attestation:verification_cc_proto",
        "@oak//proto/session:messages_cc_proto",

        # JWT-ONLY Tink Dependencies
        "@com_google_tink//tink:keyset_handle",
        "@com_google_tink//tink/config:global_registry",
        "@com_google_tink//tink/jwt:jwk_set_converter",
        "@com_google_tink//tink/jwt:jwt_public_key_verify",
        "@com_google_tink//tink/jwt:jwt_signature_config",
        "@com_google_tink//tink/jwt:jwt_validator",
        "@com_google_tink//tink/jwt:verified_jwt",
        "@com_google_tink//tink/util:statusor",  # For crypto::tink::util::StatusOr
        "@nlohmann_json//:json",  # Needed for nested claims parsing
    ],
)

# Rust library for the client session config
rust_library(
    name = "client_session_config_rs",
    srcs = ["client_session_config.rs"],
    deps = [
        "@oak//oak_session",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:anyhow",
        # For Rust-side verification
        "@oak_crates_index//:p256",
        "@oak_crates_index//:ecdsa",
    ],
)

# C++ bridge for the client session config
cc_library(
    name = "client_session_config_cc",
    hdrs = ["client_session_config.h"],
    deps = [":client_session_config_rs"],
)

# Add near other cc_library targets
cc_library(
    name = "proto_util",
    srcs = ["proto_util.cc"],
    hdrs = ["proto_util.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_protobuf//:protobuf",
    ],
)

proto_library(
    name = "test_client_proto",
    srcs = ["test_client.proto"],
)

cc_proto_library(
    name = "test_client_cc_proto",
    deps = [":test_client_proto"],
)

cc_grpc_library(
    name = "test_client_cc_grpc",
    srcs = [":test_client_proto"],
    grpc_only = True,
    deps = [":test_client_cc_proto"],
)

# A local gRPC client for testing the server
cc_binary(
    name = "test_client",
    srcs = ["test_client.cc"],
    deps = [
        ":client_session_config_cc",
        ":proto_util",
        ":session_utils",  # Added dependency
        ":test_client_cc_grpc",
        ":verifier_cc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",  # For the new flag
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/oak_session:client_session",
        "@oak//proto/services:session_v1_service_cc_grpc",
    ],
)

pkg_tar(
    name = "main_tar",
    srcs = [":main"],
    include_runfiles = True,
)

genrule(
    name = "copy_intel_jwks",
    srcs = ["@intel_jwks//file"],
    outs = ["intel_jwks.json"],
    cmd = "cp $< $@",
)

pkg_tar(
    name = "intel_jwks_layer",
    srcs = [":copy_intel_jwks"],
    mode = "0644",
    package_dir = "/etc/pki/intel",
    remap_paths = {"intel_jwks.json": "jwks.json"},
)

genrule(
    name = "copy_google_jwks",
    srcs = ["@google_jwks//file"],
    outs = ["google_jwks.json"],
    cmd = "cp $< $@",
)

pkg_tar(
    name = "google_jwks_layer",
    srcs = [":copy_google_jwks"],
    mode = "0644",
    package_dir = "/etc/pki/google",
    remap_paths = {"google_jwks.json": "jwks.json"},
)

pkg_tar(
    name = "model_layer",
    srcs = ["@gemma_weights//:gemma-3-270m-it-q4_k_m.gguf"],
    mode = "0444",
    package_dir = "/saved_model",
)

oci_image(
    name = "oci_image_ita",
    base = "@distroless_cc_debian12_base",
    cmd = [
        "--attestation_provider=ita",
        "--gpu_layers=999",
    ],
    entrypoint = ["/main"],
    env = {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:/usr/local/nvidia/lib"},
    exposed_ports = ["8000/tcp"],
    tars = [
        ":main_tar",
        ":model_layer",
    ],
)

oci_load(
    name = "tarball_ita",
    image = ":oci_image_ita",
    repo_tags = ["gcp_prototype:ita_latest"],
)

oci_image(
    name = "oci_image_gca",
    base = "@distroless_cc_debian12_base",
    cmd = [
        "--attestation_provider=gca",
        "--gpu_layers=999",
    ],
    entrypoint = ["/main"],
    env = {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:/usr/local/nvidia/lib"},
    exposed_ports = ["8000/tcp"],
    tars = [
        ":main_tar",
        ":model_layer",
    ],
)

oci_load(
    name = "tarball_gca",
    image = ":oci_image_gca",
    repo_tags = ["gcp_prototype:gca_latest"],
)

define_load_runner("ita", "ita_latest")

define_load_runner("gca", "gca_latest")

image_digest_flag(
    name = "server_digest",
    build_setting_default = "skip",
)

pkg_tar(
    name = "test_client_tar",
    srcs = [":test_client"],
    include_runfiles = True,
)

generate_policy(
    name = "gen_policy_ita",
    out = "policy_ita.textproto",
    digest_flag = ":server_digest",
    verifier_type = "ITA",
)

pkg_tar(
    name = "policy_layer_ita",
    srcs = [":gen_policy_ita"],
    mode = "0644",
    package_dir = "/etc/confidential",
    remap_paths = {"policy_ita.textproto": "policy.textproto"},
)

oci_image(
    name = "client_offline_oci_image_ita",
    base = "@distroless_cc_debian12_base",
    cmd = [
        "--server_address=10.0.2.100",
        "--jwks_path=/etc/pki/intel/jwks.json",  # Points to Intel keys
        "--policy_path=/etc/confidential/policy.textproto",
    ],
    entrypoint = ["/test_client"],
    tars = [
        ":test_client_tar",
        ":intel_jwks_layer",  # Bakes in Intel keys
        ":policy_layer_ita",
    ],
)

oci_runtime_bundle(
    name = "client_offline_bundle_ita",
    image = ":client_offline_oci_image_ita",
)

generate_policy(
    name = "gen_policy_gca",
    out = "policy_gca.textproto",
    digest_flag = ":server_digest",
    verifier_type = "GCA",
)

pkg_tar(
    name = "policy_layer_gca",
    srcs = [":gen_policy_gca"],
    mode = "0644",
    package_dir = "/etc/confidential",
    remap_paths = {"policy_gca.textproto": "policy.textproto"},
)

oci_image(
    name = "client_offline_oci_image_gca",
    base = "@distroless_cc_debian12_base",
    cmd = [
        "--server_address=10.0.2.100",
        "--jwks_path=/etc/pki/google/jwks.json",  # Points to Google keys
        "--policy_path=/etc/confidential/policy.textproto",
    ],
    entrypoint = ["/test_client"],
    tars = [
        ":test_client_tar",
        ":google_jwks_layer",  # Bakes in Google keys
        ":policy_layer_gca",
    ],
)

oci_runtime_bundle(
    name = "client_offline_bundle_gca",
    image = ":client_offline_oci_image_gca",
)

# Fill in the correct repo setup if needed
# oci_push(
#     name = "push",
#     image = ":oci_image",
#     repository = "us-docker.pkg.dev/${WORKER_PROJECT}/${WORKER_REPO}/${WORKER_IMAGE}",
#     remote_tags = ["latest"]
# )
