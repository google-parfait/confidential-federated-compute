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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//containers/program_executor_tee/program_context:__subpackages__"])

cc_library(
    name = "computation_runner",
    srcs = ["computation_runner.cc"],
    hdrs = ["computation_runner.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/confidentialcompute:tff_execution_helper",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
    ],
)

pybind_extension(
    name = "computation_runner_bindings",
    srcs = ["computation_runner_bindings.cc"],
    deps = [
        ":computation_runner",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)
