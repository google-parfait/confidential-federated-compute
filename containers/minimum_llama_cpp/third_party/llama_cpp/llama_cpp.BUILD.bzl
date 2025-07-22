"""
A custom build file for llama.cpp
"""

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["COPYING"])

COPTS = [
    "-Wno-unused-function",
    "-Wno-implicit-function-declaration",
    "-Wno-int-conversion",
    "-D_GNU_SOURCE",
]

LINKOPTS = ["-lpthread"]

cc_library(
    name = "ggml-base",
    srcs = glob([
        "ggml/src/*.c",
        "ggml/src/*.cpp",
    ]),
    hdrs = glob([
        "ggml/include/*.h",
        "ggml/src/*.h",
        "ggml/src/ggml-cpu/*.h",
    ]),
    includes = [
        "ggml/include",
        "ggml/src",
    ],
    copts = COPTS,
    local_defines = [
        "GGML_VERSION='\"0.0.0\"'",
        "GGML_COMMIT='\"6efcd65945a98cf6883cdd9de4c8ccd8c79d219a\"'",
        "GGML_USE_CPU",
        "GGML_USE_CUDA",
    ],
    linkopts = LINKOPTS,
)

cc_library(
    name = "ggml-cpu",
    srcs = glob([
        "ggml/src/ggml-cpu/*.c",
        "ggml/src/ggml-cpu/*.cpp",
        "ggml/src/ggml-cpu/amx/*.cpp",
        "ggml/src/ggml-cpu/arch/x86/*.cpp",
        "ggml/src/ggml-cpu/arch/x86/*.c",
    ]),
    hdrs = glob([
        "ggml/src/ggml-cpu/*h",
        "ggml/src/ggml-cpu/amx/*h",
    ]),
    includes = [
        "ggml/include",
        "ggml/src",
        "ggml/src/ggml-cpu",
    ],
    copts = COPTS,
    deps = [
        ":ggml-base",
    ],
    linkopts = LINKOPTS,
)

cuda_library(
    name = "ggml-cuda",
    srcs = glob([
        "ggml/src/ggml-cuda/**/*.cu",
    ]),
    hdrs = glob([
        "ggml/src/ggml-cuda/**/*.cuh",
        "ggml/src/ggml-cuda/**/*.h",
    ]),
    includes = [
        "ggml/src",
    ],
    copts = [
        "-stdlib=libstdc++",
    ],
    local_defines = [
        # Turn off vmm which requires linking to Nvidia driver.
        "GGML_CUDA_NO_VMM",
    ],
    deps = [
        ":ggml-base",
        "@cuda_curand//:curand",
        "@cuda_cublas//:cublas",
    ],
    alwayslink = True,
)

cc_library(
    name = "llama_cpp",
    srcs = glob([
        "src/*.cpp",
    ]) + [
        "common/common.cpp",
        "common/log.cpp",
        "common/ngram-cache.cpp",
        "common/regex-partial.cpp",
        "common/sampling.cpp",
        "common/speculative.cpp",
    ],
    hdrs = glob([
        "include/*.h",
        "src/*.h",
    ]) + [
        "common/base64.hpp",
        "common/common.h",
        "common/log.h",
        "common/ngram-cache.h",
        "common/regex-partial.h",
        "common/sampling.h",
        "common/speculative.h",
    ],
    includes = [
        "include",
    ],
    copts = COPTS,
    deps = [
        ":ggml-base",
        ":ggml-cpu",
        ":ggml-cuda",
    ],
)
