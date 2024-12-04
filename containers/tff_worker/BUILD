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

cc_library(
    name = "pipeline_transform_server",
    srcs = ["pipeline_transform_server.cc"],
    hdrs = ["pipeline_transform_server.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@federated-compute//fcp/base:status_converters",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_worker_configuration_cc_proto",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_parser",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/tensorflow:converters",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:federating_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:reference_resolving_executor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:tensorflow_executor",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)

cc_test(
    name = "pipeline_transform_server_test",
    size = "small",
    srcs = ["pipeline_transform_server_test.cc"],
    data = [
        "//containers/tff_worker:testing/aggregation_computation.txtpb",
        "//containers/tff_worker:testing/client_work_computation.txtpb",
    ],
    tags = [
        # There is an incompatibility between the TensorFlow dependencies for
        # this target and our ubsan configuration that causes the linker step
        # in the build to fail. When upgrading the project's TensorFlow or Bazel
        # versions, this need for this tag should be re-evaluated.
        "noubsan",
    ],
    deps = [
        ":pipeline_transform_server",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_grpc",
        "@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_cc_proto",
        "@federated-compute//fcp/protos/confidentialcompute:tff_worker_configuration_cc_proto",
        "@federated_language//federated_language/proto:computation_cc_proto",
        "@googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/testing:test_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/executors:cardinalities",
        "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_cc_proto",
    ],
)
