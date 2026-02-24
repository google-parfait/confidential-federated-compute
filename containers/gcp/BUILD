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

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_rust//rust:defs.bzl", "rust_library")
load(":build_defs.bzl", "define_load_runner", "generate_policy", "image_digest_flag")

# FUTURE WORK(b/452094015): Revise and migrate this to FCP in a revised form.
proto_library(
    name = "attestation_policy_proto",
    srcs = ["attestation_policy.proto"],
)

cc_proto_library(
    name = "attestation_policy_cc_proto",
    deps = [":attestation_policy_proto"],
)

cc_proto_library(
    name = "google_rpc_code_cc_proto",
    deps = ["@com_google_googleapis//google/rpc:code_proto"],
)

cc_proto_library(
    name = "google_rpc_status_cc_proto",
    deps = ["@com_google_googleapis//google/rpc:status_proto"],
)

cc_library(
    name = "oak_session_utils",
    srcs = ["oak_session_utils.cc"],
    hdrs = ["oak_session_utils.h"],
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

cc_library(
    name = "http_client",
    srcs = ["http_client.cc"],
    hdrs = ["http_client.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@curl",
    ],
)

cc_library(
    name = "json_parsing_utils",
    srcs = ["json_parsing_utils.cc"],
    hdrs = ["json_parsing_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nlohmann_json//:json",
    ],
)

cc_library(
    name = "proto_parsing_utils",
    srcs = ["proto_parsing_utils.cc"],
    hdrs = ["proto_parsing_utils.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_protobuf//:protobuf",
    ],
)

# FUTURE WORK(b/452094015): Potentially migrate generic batched inference
# targets to the batched_inference directory in the main part of the CFC repo.
proto_library(
    name = "batched_inference_proto",
    srcs = ["batched_inference.proto"],
    deps = [
        "@com_google_googleapis//google/rpc:status_proto",
    ],
)

cc_proto_library(
    name = "batched_inference_cc_proto",
    deps = [":batched_inference_proto"],
)

cc_library(
    name = "batched_inference_engine",
    srcs = [],
    hdrs = ["batched_inference_engine.h"],
    deps = [
        ":batched_inference_cc_proto",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "llama_cpp_batched_inference_engine",
    srcs = ["llama_cpp_batched_inference_engine.cc"],
    hdrs = ["llama_cpp_batched_inference_engine.h"],
    deps = [
        ":batched_inference_cc_proto",
        ":batched_inference_engine",
        ":google_rpc_code_cc_proto",
        ":google_rpc_status_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llama_cpp",
    ],
)

cc_library(
    name = "attestation_token_provider_cc",
    srcs = ["attestation_token_provider.cc"],
    hdrs = ["attestation_token_provider.h"],
    deps = [
        ":http_client",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "attestation_token_verifier_cc",
    srcs = ["attestation_token_verifier.cc"],
    hdrs = ["attestation_token_verifier.h"],
    deps = [
        ":attestation_policy_cc_proto",
        ":http_client",
        ":json_parsing_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_tink//tink:keyset_handle",
        "@com_google_tink//tink/config:global_registry",
        "@com_google_tink//tink/jwt:jwk_set_converter",
        "@com_google_tink//tink/jwt:jwt_public_key_verify",
        "@com_google_tink//tink/jwt:jwt_signature_config",
        "@com_google_tink//tink/jwt:jwt_validator",
        "@com_google_tink//tink/jwt:verified_jwt",
        "@com_google_tink//tink/util:statusor",
        "@nlohmann_json//:json",
        "@oak//cc/attestation/verification:attestation_verifier",
        "@oak//proto/attestation:endorsement_cc_proto",
        "@oak//proto/attestation:evidence_cc_proto",
        "@oak//proto/attestation:verification_cc_proto",
        "@oak//proto/session:messages_cc_proto",
    ],
)

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

cc_library(
    name = "server_session_config_cc",
    hdrs = ["server_session_config.h"],
    deps = [":server_session_config_rs"],
)

rust_library(
    name = "client_session_config_rs",
    srcs = ["client_session_config.rs"],
    deps = [
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:ecdsa",
        "@oak_crates_index//:p256",
    ],
)

cc_library(
    name = "client_session_config_cc",
    hdrs = ["client_session_config.h"],
    deps = [":client_session_config_rs"],
)

cc_library(
    name = "server",
    srcs = ["server.cc"],
    hdrs = ["server.h"],
    deps = [
        ":attestation_token_provider_cc",
        ":oak_session_utils",
        ":server_session_config_cc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_tink//tink/config:tink_config",
        "@com_google_tink//tink/signature:signature_config",
        "@oak//cc/ffi:rust_bytes",
        "@oak//cc/oak_session:server_session",
        "@oak//proto/services:session_v1_service_cc_grpc",
    ],
)

cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    deps = [
        ":attestation_token_verifier_cc",
        ":client_session_config_cc",
        ":oak_session_utils",
        ":proto_parsing_utils",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/oak_session:client_session",
        "@oak//proto/services:session_v1_service_cc_grpc",
    ],
)

cc_binary(
    name = "batched_inference_gcp_main",
    srcs = ["batched_inference_gcp_main.cc"],
    deps = [
        ":attestation_token_provider_cc",
        ":batched_inference_cc_proto",
        ":batched_inference_engine",
        ":llama_cpp_batched_inference_engine",
        ":server",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

pkg_tar(
    name = "batched_inference_gcp_tar",
    srcs = [":batched_inference_gcp_main"],
    include_runfiles = True,
)

pkg_tar(
    name = "model_layer",
    srcs = ["@gemma_weights//:gemma-3-270m-it-q4_k_m.gguf"],
    mode = "0444",
    package_dir = "/saved_model",
)

oci_image(
    name = "batched_inference_gcp_ita_oci_image",
    base = "@distroless_cc_debian12_base",
    cmd = [
        "--attestation_provider=ita",
        "--gpu_layers=999",
    ],
    entrypoint = ["/batched_inference_gcp_main"],
    env = {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:/usr/local/nvidia/lib"},
    exposed_ports = ["8000/tcp"],
    tars = [
        ":batched_inference_gcp_tar",
        ":model_layer",
    ],
)

oci_image(
    name = "batched_inference_gcp_gca_oci_image",
    base = "@distroless_cc_debian12_base",
    cmd = [
        "--attestation_provider=gca",
        "--gpu_layers=999",
    ],
    entrypoint = ["/batched_inference_gcp_main"],
    env = {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:/usr/local/nvidia/lib"},
    exposed_ports = ["8000/tcp"],
    tars = [
        ":batched_inference_gcp_tar",
        ":model_layer",
    ],
)

oci_load(
    name = "batched_inference_gcp_ita_tarball",
    image = ":batched_inference_gcp_ita_oci_image",
    repo_tags = ["batched_inference_gcp:ita_latest"],
)

oci_load(
    name = "batched_inference_gcp_gca_tarball",
    image = ":batched_inference_gcp_gca_oci_image",
    repo_tags = ["batched_inference_gcp:gca_latest"],
)

define_load_runner("batched_inference_gcp_ita", "batched_inference_gcp:ita_latest")

define_load_runner("batched_inference_gcp_gca", "batched_inference_gcp:gca_latest")

cc_binary(
    name = "batched_inference_oak_main",
    srcs = ["batched_inference_oak_main.cc"],
    deps = [
        ":attestation_token_verifier_cc",
        ":batched_inference_cc_proto",
        ":client",
        ":proto_parsing_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@confidential-federated-compute//containers/batched_inference:batched_inference_provider",
        "@confidential-federated-compute//containers/batched_inference:batched_inference_server",
        "@oak//cc/containers/sdk:encryption_key_handle",
        "@oak//cc/containers/sdk:orchestrator_client",
        "@oak//cc/containers/sdk:signing_key_handle",
    ],
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

image_digest_flag(
    name = "server_digest",
    build_setting_default = "skip",
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

pkg_tar(
    name = "batched_inference_oak_tar",
    srcs = [":batched_inference_oak_main"],
    include_runfiles = True,
)

oci_image(
    name = "batched_inference_oak_gca_oci_image",
    base = "@distroless_cc_debian12_base",
    cmd = [
        "--server_address=10.0.2.100",
        "--jwks_path=/etc/pki/google/jwks.json",
        "--policy_path=/etc/confidential/policy.textproto",
    ],
    entrypoint = ["/batched_inference_oak_main"],
    tars = [
        ":batched_inference_oak_tar",
        ":google_jwks_layer",
        ":policy_layer_gca",
    ],
)

oci_image(
    name = "batched_inference_oak_ita_oci_image",
    base = "@distroless_cc_debian12_base",
    cmd = [
        "--server_address=10.0.2.100",
        "--jwks_path=/etc/pki/intel/jwks.json",
        "--policy_path=/etc/confidential/policy.textproto",
        "--dump_jwt=true",
    ],
    entrypoint = ["/batched_inference_oak_main"],
    tars = [
        ":batched_inference_oak_tar",
        ":intel_jwks_layer",
        ":policy_layer_ita",
    ],
)

oci_runtime_bundle(
    name = "batched_inference_oak_gca_oci_runtime_bundle",
    image = ":batched_inference_oak_gca_oci_image",
)

oci_runtime_bundle(
    name = "batched_inference_oak_ita_oci_runtime_bundle",
    image = ":batched_inference_oak_ita_oci_image",
)
