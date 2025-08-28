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

"""This file contains custom Bazel rules for llama.cpp."""

load("@rules_cc//cc:defs.bzl", "cc_library")

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
    hdrs = glob(
        [
            "ggml/include/*.h",
            "ggml/src/*.h",
            "ggml/src/ggml-cpu/*.h",
        ],
    ),
    includes = [
        "ggml/include",
        "ggml/src",
    ],
    copts = COPTS,
    local_defines = [
        "GGML_VERSION='\"0.0.0\"'",
        "GGML_COMMIT='\"6efcd65945a98cf6883cdd9de4c8ccd8c79d219a\"'",
        "GGML_USE_CPU",
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
        "common/build-info.cpp",
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
    local_defines = [
        "LLAMA_BUILD_NUMBER_DEFINED=1",
        "LLAMA_COMMIT_DEFINED='\"be48528b\"'",
        "LLAMA_COMPILER_DEFINED='\"bazel\"'",
        "LLAMA_BUILD_TARGET_DEFINED='\"unknown\"'",
    ],
    deps = [
        ":ggml-base",
        ":ggml-cpu",
    ],
)
