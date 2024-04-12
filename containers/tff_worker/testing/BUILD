# Copyright 2023 Google LLC.
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
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//:__subpackages__"])

exports_files([
    "aggregation_computation.txtpb",
    "client_work_computation.txtpb",
])

py_library(
    name = "test_computations",
    testonly = True,
    srcs = ["test_computations.py"],
    deps = [
        "@pypi_deps_tensorflow//:pkg",
        "@pypi_deps_tensorflow_federated//:pkg",
    ],
)

py_binary(
    name = "generate_test_computations",
    testonly = True,
    srcs = ["generate_test_computations.py"],
    deps = [
        ":test_computations",
        "@pypi_deps_tensorflow_federated//:pkg",
    ],
)
