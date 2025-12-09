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
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(
    default_visibility = ["//:__subpackages__"],
    features = ["layering_check"],
)

cc_library(
    name = "fns",
    srcs = [
        "confidential_transform_server.cc",
        "map_fn.cc",
    ],
    hdrs = [
        "confidential_transform_server.h",
        "fn.h",
        "fn_factory.h",
        "map_fn.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//containers:confidential_transform_server_base",
        "//containers:session",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:any_cc_proto",
        "@federated-compute//fcp/base:base",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
    ],
)

cc_test(
    name = "confidential_transform_server_test",
    srcs = ["confidential_transform_server_test.cc"],
    deps = [
        ":fns",
        "//containers:crypto_test_utils",
        "//containers:session",
        "//testing:matchers",
        "//testing:parse_text_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:any_cc_proto",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:kms_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@oak//cc/crypto:client_encryptor",
        "@oak//cc/crypto:encryption_key",
    ],
)

cc_test(
    name = "map_fn_test",
    srcs = ["map_fn_test.cc"],
    deps = [
        ":fns",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_protobuf//:any_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tee_payload_metadata_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
