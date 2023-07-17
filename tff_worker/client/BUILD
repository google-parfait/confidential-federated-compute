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
load("@rules_proto_grpc//cpp:defs.bzl", "cpp_grpc_library")
load("@rules_proto_grpc//cpp:defs.bzl", "cpp_proto_library")
package(default_visibility = ["//:__subpackages__"])

# TODO: Use the C++ proto and gRPC targets defined in FCP rather than adding
# separate targets here.
cpp_proto_library(
    name = "pipeline_transform_cpp_proto",
    protos = ["@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_proto"],
)

cpp_grpc_library(
    name = "pipeline_transform_cpp_grpc",
    protos = ["@federated-compute//fcp/protos/confidentialcompute:pipeline_transform_proto"],
)

cc_binary(
    name = "pipeline_transform_cpp_client",
    srcs = ["pipeline_transform_client.cc"],
    deps = [
        ":pipeline_transform_cpp_grpc",
        ":pipeline_transform_cpp_proto",
    ],
)
