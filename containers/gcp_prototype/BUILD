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
load(":build_defs.bzl", "generate_policy", "image_digest_flag")

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

# The gRPC server that will run in the container
cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
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

# A local gRPC client for testing the server
cc_binary(
    name = "test_client",
    srcs = ["test_client.cc"],
    deps = [
        ":client_session_config_cc",
        ":proto_util",
        ":session_utils",  # Added dependency
        ":verifier_cc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",  # For the new flag
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@oak//cc/oak_session:client_session",
        "@oak//proto/services:session_v1_service_cc_grpc",
    ],
)

pkg_tar(
    name = "main_tar",
    srcs = [":main"],
    include_runfiles = True,
)

oci_image(
    name = "oci_image",
    base = "@distroless_cc_debian12_base",
    cmd = ["--attestation_provider=ita"],
    entrypoint = ["/main"],
    exposed_ports = ["8000/tcp"],
    tars = [
        ":main_tar",
    ],
)

# oci_load rule generates the tarball.sh script
oci_load(
    name = "tarball",
    image = ":oci_image",
    repo_tags = [
        "gcp_prototype:latest",
    ],
)

pkg_tar(
    name = "test_client_tar",
    srcs = [":test_client"],
    include_runfiles = True,
)

# 1. Copy the downloaded file to a local name we control
genrule(
    name = "copy_jwks",
    srcs = ["@intel_jwks//file"],
    outs = ["jwks.json"],
    cmd = "cp $< $@",
)

# 2. Tar it up, placing it in the desired directory
pkg_tar(
    name = "intel_jwks_layer",
    srcs = [":copy_jwks"],
    mode = "0644",
    package_dir = "/etc/pki/intel",
)

# Default to "skip" so it doesn't fail if you forget it during local dev.
image_digest_flag(
    name = "server_digest",
    build_setting_default = "skip",
)

# Generates the runtime policy file based on build configuration.
generate_policy(
    name = "gen_offline_policy",
    out = "offline_policy.textproto",
    digest_flag = ":server_digest",
)

# Packages the generated policy into /etc/confidential/policy.textproto
# for inclusion in the client container.
pkg_tar(
    name = "policy_layer",
    srcs = [":gen_offline_policy"],
    mode = "0644",
    package_dir = "/etc/confidential",
    remap_paths = {"offline_policy.textproto": "policy.textproto"},
)

# Dedicated image for TEE testing that bakes in the necessary flags
oci_image(
    name = "client_offline_oci_image",
    base = "@distroless_cc_debian12_base",
    cmd = [
        "--server_address=10.0.2.100",
        "--jwks_path=/etc/pki/intel/jwks.json",
        "--policy_path=/etc/confidential/policy.textproto",
    ],
    entrypoint = ["/test_client"],
    tars = [
        ":test_client_tar",
        ":intel_jwks_layer",
        ":policy_layer",
    ],
)

oci_runtime_bundle(
    name = "client_offline_bundle",
    image = ":client_offline_oci_image",
)

# Genrule loads image, then inspects via Docker CLI
genrule(
    name = "load_and_print_digest_runner",
    outs = ["load_and_print_digest.sh"],  # Output a dummy script
    cmd = """
        set -e # Exit on error
        TARBALL_SCRIPT=$(location :tarball)
        IMAGE_TAG="gcp_prototype:latest" # Must match repo_tags in :tarball

        # --- STEP 1: Execute the oci_load script ---
        echo "Executing oci_load script: $$TARBALL_SCRIPT" >&2
        $$TARBALL_SCRIPT # This runs 'docker load'
        echo "Image load complete." >&2

        # --- STEP 2: Get the Image ID for the loaded tag ---
        echo "Inspecting tag '$$IMAGE_TAG' to get Image ID..." >&2
        LOADED_ID=$$(docker image inspect "$$IMAGE_TAG" --format '{{.Id}}' 2>/dev/null)

        if [ -z "$$LOADED_ID" ]; then
            echo "ERROR: Could not inspect tag '$$IMAGE_TAG' to get Image ID after load." >&2
            exit 1
        fi
        echo "Found Image ID for tag '$$IMAGE_TAG': $$LOADED_ID" >&2

        # --- STEP 3: Inspect the specific Image ID to get the RepoDigest (Manifest Digest) ---
        echo "Inspecting loaded Image ID '$$LOADED_ID' for RepoDigest..." >&2
        # Inspect the specific ID and format RepoDigests.
        # Use grep/head to robustly get the first sha256:... digest listed.
        DIGEST=$$(docker image inspect "$$LOADED_ID" --format '{{range .RepoDigests}}{{.}}{{println}}{{end}}' 2>/dev/null | grep -o 'sha256:[a-f0-9]*' | head -n 1 || true)

        # Fallback to the Image ID itself ONLY if RepoDigests is empty for that ID
        if [ -z "$$DIGEST" ]; then
            echo "WARNING: RepoDigest not found for specific Image ID $$LOADED_ID after load. Falling back to Image ID (Config Digest)." >&2
            # Use the ID we got from inspecting the tag
            DIGEST="$$LOADED_ID"
        else
            echo "Found RepoDigest (Manifest Digest) for Image ID $$LOADED_ID: $$DIGEST" >&2
        fi

        # --- STEP 4: Print the final digest ---
        # This should now be the Manifest Digest if available, otherwise the Config Digest.
        echo "==================================================" >&2
        echo "Server Image Docker Digest: $$DIGEST" >&2
        echo "==================================================" >&2

        # --- STEP 5: Generate the dummy output script for bazel run ---
        printf '#!/bin/bash\\n echo "Wrapper script finished."\\n exit 0\\n' > $@
        chmod +x $@
    """,
    executable = True,  # Make runnable via `bazel run`
    local = True,  # Allow access to local Docker daemon
    tools = [":tarball"],  # Need the tarball script path
    visibility = ["//visibility:public"],
)

# Fill in the correct repo setup if needed
# oci_push(
#     name = "push",
#     image = ":oci_image",
#     repository = "us-docker.pkg.dev/${WORKER_PROJECT}/${WORKER_REPO}/${WORKER_IMAGE}",
#     remote_tags = ["latest"]
# )
