load("@rules_cc//cc:defs.bzl", "cc_library")

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "libcppbor",
    srcs = glob(["src/*.cpp"]),
    hdrs = glob(["include/cppbor/*.h"]),
    include_prefix = "libcppbor",
    includes = ["include/cppbor"],
    local_defines = ["__TRUSTY__"],
    visibility = ["//visibility:public"],
    deps = ["@boringssl//:crypto"],
)
