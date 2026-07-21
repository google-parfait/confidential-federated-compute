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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//containers:__subpackages__"])

cc_library(
    name = "interval",
    hdrs = ["interval.h"],
    deps = [
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "interval_set",
    hdrs = ["interval_set.h"],
    deps = [
        ":interval",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:check",
    ],
)

cc_test(
    name = "interval_set_test",
    size = "small",
    srcs = ["interval_set_test.cc"],
    deps = [
        ":interval_set",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "interval_map",
    hdrs = ["interval_map.h"],
    deps = [
        ":interval",
        ":interval_set",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:check",
    ],
)

cc_test(
    name = "interval_map_test",
    size = "small",
    srcs = ["interval_map_test.cc"],
    deps = [
        ":interval",
        ":interval_map",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
