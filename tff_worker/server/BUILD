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
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//:__subpackages__"])

py_binary(
    name = "pipeline_transform_server",
    srcs = ["pipeline_transform_server.py"],
    main = "pipeline_transform_server.py",
    # TFF protos are not included in the version of TFF released as a python
    # package so we import them separately from a git_repository rule.
    deps = [
        ":tff_transforms",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_py_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_py_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_worker_configuration_py_proto",
        "@oak//oak_containers/proto:interfaces_py_grpc",
        "@pypi_deps_tensorflow_federated//:pkg",
        "@tensorflow-federated//tensorflow_federated/proto/v0:computation_py_pb2",
        "@tensorflow-federated//tensorflow_federated/proto/v0:executor_py_pb2",
    ],
)

filegroup(
    name = "pipeline_transform_server_zip",
    srcs = [":pipeline_transform_server"],
    output_group = "python_zip_file",
    tags = ["manual"],
)

py_test(
    name = "pipeline_transform_server_test",
    srcs = ["pipeline_transform_server_test.py"],
    tags = ["no-remote-exec"],  # https://github.com/bazelbuild/rules_python/issues/571
    deps = [
        ":pipeline_transform_server",
        ":tff_transforms",
        "//tff_worker/server/testing:checkpoint_test_utils",
        "//tff_worker/server/testing:test_computations",
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
    tags = ["no-remote-exec"],  # https://github.com/bazelbuild/rules_python/issues/571
    deps = [
        ":tff_transforms",
        "//tff_worker/server/testing:checkpoint_test_utils",
        "@federated-compute//fcp/protos/confidentialcompute:tff_worker_configuration_py_proto",
        "@org_tensorflow//tensorflow/core:protos_all_py",
        "@pypi_deps_tensorflow_federated//:pkg",
        "@tensorflow-federated//tensorflow_federated/proto/v0:computation_py_pb2",
        "@tensorflow-federated//tensorflow_federated/proto/v0:executor_py_pb2",
    ],
)

cc_library(
    name = "cpp_pipeline_transform_server",
    srcs = ["pipeline_transform_server.cc"],
    hdrs = ["pipeline_transform_server.h"],
    tags = ["manual"],  # TF compilation is causing CI timeouts
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_parser",
        "@federated-compute//fcp/aggregation/tensorflow:converters",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_worker_configuration_cc_proto",
        "@federated-compute//fcp/base:status_converters",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@tensorflow-federated//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "@tensorflow-federated//tensorflow_federated/cc/core/impl/executors:executor",
        "@tensorflow-federated//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "@tensorflow-federated//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "@tensorflow-federated//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
        "@tensorflow-federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_test(
    name = "cpp_pipeline_transform_server_test",
    size = "small",
    srcs = ["pipeline_transform_server_test.cc"],
    data = [
        "//tff_worker/server/testing:aggregation_computation.pbtxt",
        "//tff_worker/server/testing:client_work_computation.pbtxt",
    ],
    tags = ["manual"],  # TF compilation is causing CI timeouts.
    deps = [
        ":cpp_pipeline_transform_server",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/aggregation/core:tensor",
        "@federated-compute//fcp/aggregation/protocol:federated_compute_checkpoint_builder",
        "@federated-compute//fcp/aggregation/testing:test_data",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_worker_configuration_cc_proto",
        "@googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@tensorflow-federated//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "@tensorflow-federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
    # Static registration needs always_link to avoid accidental removal by the linker.
    linkstatic=True,
)

