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
    name = "budget_interval_map",
    srcs = ["budget_interval_map.cc"],
    hdrs = ["budget_interval_map.h"],
    deps = [
        "//containers/fed_sql:interval",
        "//containers/fed_sql:interval_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

cc_test(
    name = "budget_interval_map_test",
    size = "small",
    srcs = ["budget_interval_map_test.cc"],
    deps = [
        ":budget_interval_map",
        "//containers/fed_sql:interval",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "time_budget",
    srcs = ["time_budget.cc"],
    hdrs = ["time_budget.h"],
    deps = [
        ":budget_interval_map",
        "//containers/fed_sql:budget_cc_proto",
        "//containers/fed_sql:interval",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "time_budget_test",
    srcs = ["time_budget_test.cc"],
    deps = [
        ":time_budget",
        "//containers/fed_sql:budget_cc_proto",
        "//containers/fed_sql:interval",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)
