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
    name = "map_fn",
    srcs = ["map_fn.cc"],
    hdrs = ["map_fn.h"],
    deps = [
        "//containers:session",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:any_cc_proto",
        "@federated-compute//fcp/base:base",
        "@federated-compute//fcp/protos/confidentialcompute:tee_payload_metadata_cc_proto",
    ],
)

cc_test(
    name = "map_fn_test",
    srcs = ["map_fn_test.cc"],
    deps = [
        ":map_fn",
        "//testing:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_protobuf//:any_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tee_payload_metadata_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
