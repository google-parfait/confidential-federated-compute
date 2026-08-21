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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(
    default_visibility = ["//:__subpackages__"],
    features = ["layering_check"],
)

cc_library(
    name = "privacy_id_utils",
    hdrs = ["privacy_id_utils.h"],
    deps = [
        "//containers:big_endian",
        "@com_google_absl//absl/strings",
        "@federated_compute//fcp/base:digest",
    ],
)

cc_library(
    name = "utf8_utils",
    srcs = ["utf8_utils.cc"],
    hdrs = ["utf8_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        # Note: Depending directly on protobuf's internal third_party/utf8_range
        # may be fragile if upstream changes its export structure, but it is the
        # easiest option available without introducing major new dependencies.
        "@com_google_protobuf//third_party/utf8_range:utf8_validity",
    ],
)

cc_test(
    name = "utf8_utils_test",
    size = "small",
    srcs = ["utf8_utils_test.cc"],
    deps = [
        ":utf8_utils",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
