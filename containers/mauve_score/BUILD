# Copyright 2026 Google LLC.
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

"""BUILD rules for the MAUVE score container."""

load("@bazel_toolchains//rules/exec_properties:exec_properties.bzl", "create_rbe_exec_properties_dict")
load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

platform(
    name = "remote_platform",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    exec_properties = create_rbe_exec_properties_dict(
        container_image = "docker://gcr.io/cloud-marketplace/google/debian12@sha256:be15f84c44c4ee68b4f187128f0278df1b0f424c04fb5f08b098967d5b896388",
        os_family = "Linux",
    ),
    parents = ["@local_config_platform//:host"],
)

compile_pip_requirements(
    name = "requirements",
    timeout = "long",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock.txt",
)

py_library(
    name = "compute_mauve_score_lib",
    srcs = ["compute_mauve_score_lib.py"],
    deps = [
        requirement("numpy"),
        requirement("scikit-learn"),
        "@federated-compute//fcp/protos/confidentialcompute:mauve_score_config_py_pb2",
    ],
)

py_test(
    name = "compute_mauve_score_lib_test",
    srcs = ["compute_mauve_score_lib_test.py"],
    deps = [
        ":compute_mauve_score_lib",
        "@federated-compute//fcp/protos/confidentialcompute:mauve_score_config_py_pb2",
    ],
)
