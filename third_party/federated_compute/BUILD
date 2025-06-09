load("@rules_rust//cargo:defs.bzl", "cargo_build_script")
load("@rules_rust//rust:defs.bzl", "rust_library")

cargo_build_script(
    name = "build",
    srcs = ["build.rs"],
    build_script_env = {
        "PROTOC": "$(execpath @com_google_protobuf//:protoc)",
        "DESCRIPTOR_PROTO": "$(location @com_google_protobuf//:descriptor_proto_srcs)",
        "EVIDENCE_PROTO": "$(location @oak//proto/attestation:evidence.proto)",
        "LEDGER_PROTO": "$(location @federated-compute//fcp/protos/confidentialcompute:ledger.proto)",
        "DATETIME_PROTO": "$(location @com_google_googleapis//google/type:datetime.proto)",
    },
    data = [
        "@com_google_googleapis//google/type:datetime.proto",
        "@com_google_protobuf//:descriptor_proto_srcs",
        "@com_google_protobuf//:well_known_type_protos",
        "@federated-compute//fcp/client:event_time_range.proto",
        "@federated-compute//fcp/protos/confidentialcompute:access_policy.proto",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header.proto",
        "@federated-compute//fcp/protos/confidentialcompute:ledger.proto",
        "@federated-compute//fcp/protos/confidentialcompute:payload_metadata.proto",
        "@federated-compute//fcp/protos/confidentialcompute:verification_record.proto",
        "@oak//proto:digest.proto",
        "@oak//proto:variant.proto",
        "@oak//proto/attestation:endorsement.proto",
        "@oak//proto/attestation:eventlog.proto",
        "@oak//proto/attestation:evidence.proto",
        "@oak//proto/attestation:reference_value.proto",
        "@oak//proto/attestation:tcb_version.proto",
        "@oak//proto/crypto:certificate.proto",
        "@oak//proto/micro_rpc:options.proto",
    ],
    tools = ["@com_google_protobuf//:protoc"],
    deps = [
        "@oak//micro_rpc_build",
        "@oak//oak_proto_build_utils",
    ],
)

rust_library(
    name = "federated_compute",
    srcs = glob(["src/**"]),
    proc_macro_deps = ["@oak_crates_index//:prost-derive"],
    tags = ["no-clippy"],
    visibility = ["//visibility:public"],
    deps = [
        ":build",
        "@oak//micro_rpc",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:prost-types",
    ],
)
