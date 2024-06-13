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

package(default_visibility = ["//containers:__subpackages__"])

cc_library(
    name = "crypto",
    srcs = ["crypto.cc"],
    hdrs = ["crypto.h"],
    deps = [
        "@boringssl//:crypto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base:compression",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/confidentialcompute:cose",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:blob_header_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@oak//proto/containers:orchestrator_crypto_cc_grpc",
        "@oak//proto/containers:orchestrator_crypto_cc_proto",
    ],
)

cc_library(
    name = "session",
    srcs = ["session.cc"],
    hdrs = ["session.h"],
    deps = [
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "session_test",
    size = "small",
    srcs = ["session_test.cc"],
    deps = [
        ":session",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "blob_metadata",
    srcs = ["blob_metadata.cc"],
    hdrs = ["blob_metadata.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/confidentialcompute:cose",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
    ],
)

cc_test(
    name = "blob_metadata_test",
    size = "small",
    srcs = ["blob_metadata_test.cc"],
    deps = [
        ":blob_metadata",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/confidentialcompute:cose",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "crypto_test_utils",
    testonly = True,
    srcs = ["crypto_test_utils.cc"],
    hdrs = ["crypto_test_utils.h"],
    deps = [
        ":crypto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/confidentialcompute:cose",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "oak_orchestrator_client",
    srcs = ["oak_orchestrator_client.cc"],
    hdrs = ["oak_orchestrator_client.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@oak//proto/containers:interfaces_cc_grpc",
        "@oak//proto/containers:interfaces_cc_proto",
    ],
)

cc_test(
    name = "crypto_test",
    size = "small",
    srcs = ["crypto_test.cc"],
    deps = [
        ":blob_metadata",
        ":crypto",
        ":crypto_test_utils",
        ":session",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/base",
        "@federated-compute//fcp/base:compression",
        "@federated-compute//fcp/confidentialcompute:crypto",
        "@googletest//:gtest_main",
        "@oak//proto/containers:orchestrator_crypto_cc_grpc",
        "@oak//proto/containers:orchestrator_crypto_cc_proto",
    ],
)

cc_test(
    name = "oak_orchestrator_client_test",
    size = "small",
    srcs = ["oak_orchestrator_client_test.cc"],
    deps = [
        ":oak_orchestrator_client",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@googletest//:gtest_main",
        "@oak//proto/containers:interfaces_cc_grpc",
        "@oak//proto/containers:interfaces_cc_proto",
    ],
)
