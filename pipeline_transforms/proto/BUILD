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
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc//cpp:defs.bzl", "cpp_grpc_library")
load("@rules_proto_grpc//cpp:defs.bzl", "cpp_proto_library")
load("@rules_proto_grpc//python:defs.bzl", "python_grpc_library")
load("@rules_proto_grpc//python:defs.bzl", "python_proto_library")

package(default_visibility = ["//:__subpackages__"])

proto_library(
    name = "pipeline_transform_proto",
    srcs = ["pipeline_transform.proto"],
    deps = ["@com_google_protobuf//:any_proto"],
)

python_grpc_library(
    name = "pipeline_transform_py_grpc",
    protos = [":pipeline_transform_proto"],
)

python_proto_library(
    name = "pipeline_transform_py_proto",
    protos = [":pipeline_transform_proto"],
)

cpp_proto_library(
    name = "pipeline_transform_cpp_proto",
    protos = [":pipeline_transform_proto"],
)

cpp_grpc_library(
    name = "pipeline_transform_cpp_grpc",
    protos = [":pipeline_transform_proto"],
)
