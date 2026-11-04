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
        "GGML_COMMIT='\"1f368f354d9edcfea9fd6a1e0989b3e7335a050f\"'",
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

# Replace CMake tokens in common/build-info.cpp.in so common/common.cpp links
# build metadata functions without a patch file.
genrule(
    name = "build_info",
    srcs = ["common/build-info.cpp.in"],
    outs = ["common/build-info.cpp"],
    cmd = (
        "sed" +
        " -e 's/@LLAMA_BUILD_NUMBER@/10400/g'" +
        " -e 's/@LLAMA_BUILD_COMMIT@/1f368f35/g'" +
        " -e 's/@BUILD_COMPILER@/bazel/g'" +
        " -e 's/@BUILD_TARGET@/linux-x86_64/g'" +
        " $(execpath common/build-info.cpp.in) > $@"
    ),
)

# Compile build-info.cpp in a separate target so -DLLAMA_COMMIT on :llama_cpp
# does not overwrite the LLAMA_COMMIT variable name in build-info.cpp.
cc_library(
    name = "llama_build_info",
    srcs = [":build_info"],
    hdrs = ["common/build-info.h"],
    includes = ["common"],
    copts = COPTS,
)

cc_library(
    name = "llama_cpp",
    srcs = glob(
        [
            "src/**/*.cpp",
            "common/*.cpp",
        ],
        exclude = [
            # Exclude optional CLI/server helpers that require external
            # libraries (nlohmann_json, libcurl, llguidance) not linked in TEE
            # containers.
            "common/arg.cpp",
            "common/chat.cpp",
            "common/chat-auto-parser-generator.cpp",
            "common/chat-auto-parser-helpers.cpp",
            "common/chat-diff-analyzer.cpp",
            "common/chat-peg-parser.cpp",
            "common/console.cpp",
            "common/download.cpp",
            "common/hf-cache.cpp",
            "common/json-partial.cpp",
            "common/json-schema-to-grammar.cpp",
            "common/llguidance.cpp",
            "common/peg-parser.cpp",
            "common/preset.cpp",
        ],
    ),
    hdrs = glob([
        "include/*.h",
        "src/**/*.h",
        "common/*.h",
        "common/*.hpp",
    ]),
    includes = [
        "common",
        "include",
        "src",
    ],
    copts = COPTS,
    local_defines = [
        "LLAMA_VERSION='\"0.1.0-dev\"'",
        "LLAMA_COMMIT='\"1f368f35\"'",
    ],
    deps = [
        ":ggml-base",
        ":ggml-cpu",
        ":llama_build_info",
    ],
)
