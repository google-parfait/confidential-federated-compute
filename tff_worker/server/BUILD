# Copyright 2023 The Confidential Federated Compute Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//:__subpackages__"])

py_binary(
    name = "pipeline_transform_server",
    srcs = ["pipeline_transform_server.py"],
    main = "pipeline_transform_server.py",
    # TFF protos are not included in the version of TFF released as a python
    # package so we import them separately from a git_repository rule.
    deps = [
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_worker_configuration_py_proto",
        "@pypi_deps_tensorflow_federated//:pkg",
        "@tensorflow-federated//tensorflow_federated/proto/v0:computation_py_pb2",
        "@tensorflow-federated//tensorflow_federated/proto/v0:executor_py_pb2",
    ],
)

py_test(
    name = "pipeline_transform_server_test",
    srcs = ["pipeline_transform_server_test.py"],
    deps = [
        ":pipeline_transform_server",
        ":tff_transforms",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_worker_configuration_py_proto",
        "@pypi_deps_portpicker//:pkg",
        "@pypi_deps_tensorflow_federated//:pkg",
    ],
)

py_library(
    name = "tff_transforms",
    srcs = ["tff_transforms.py"],
    deps = [
        "@pypi_deps_tensorflow_federated//:pkg",
        "@tensorflow-federated//tensorflow_federated/proto/v0:computation_py_pb2",
        "@tensorflow-federated//tensorflow_federated/proto/v0:executor_py_pb2",
    ],
)

py_test(
    name = "tff_transforms_test",
    srcs = ["tff_transforms_test.py"],
    deps = [
        ":tff_transforms",
        "@pypi_deps_tensorflow_federated//:pkg",
        "@tensorflow-federated//tensorflow_federated/proto/v0:computation_py_pb2",
        "@tensorflow-federated//tensorflow_federated/proto/v0:executor_py_pb2",
    ],
)
