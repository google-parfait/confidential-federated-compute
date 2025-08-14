"""BUILD file for gemma_pytorch"""

load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library")

filegroup(
    name = "tokenizer_model",
    srcs = glob(["tokenizer/*"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "gemma",
    srcs = glob(["gemma/**/*.py"]),
    imports = ["."],
    deps = [
        requirement("absl_py"),
        requirement("numpy"),
        requirement("pillow"),
        requirement("sentencepiece"),
        requirement("torch"),
    ],
    visibility = ["//visibility:public"],
)
