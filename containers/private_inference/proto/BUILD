load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "private_aratea_service_proto",
    srcs = ["private_aratea_service.proto"],
    deps = [
        "@com_google_protobuf//:struct_proto",
    ],
)

cc_proto_library(
    name = "private_aratea_service_cc_proto",
    deps = [":private_aratea_service_proto"],
)

proto_library(
    name = "verification_keys_proto",
    srcs = ["verification_keys.proto"],
    deps = ["@com_google_protobuf//:timestamp_proto"],
)

cc_proto_library(
    name = "verification_keys_cc_proto",
    deps = [":verification_keys_proto"],
)
