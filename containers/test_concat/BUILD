# Copyright 2024 Google LLC.
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

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":pipeline_transform_server",
        "//containers:oak_orchestrator_client",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@oak//oak_containers/proto:interfaces_cc_grpc",
        "@oak//oak_containers/proto:interfaces_cc_proto",
    ],
)

cc_library(
    name = "pipeline_transform_server",
    srcs = ["pipeline_transform_server.cc"],
    hdrs = ["pipeline_transform_server.h"],
    deps = [
        "//containers:crypto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
    ],
)

cc_test(
    name = "pipeline_transform_server_test",
    size = "small",
    srcs = ["pipeline_transform_server_test.cc"],
    deps = [
        ":pipeline_transform_server",
        "//containers:crypto",
        "//containers:crypto_test_utils",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@googletest//:gtest_main",
    ],
)
