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

"""Repository rule for stubbing out tf_proto_library dependencies."""

def _stub_tf_proto_library_impl(rctx):
    rctx.file("tensorflow/core/platform/BUILD", executable = False)
    rctx.file(
        "tensorflow/core/platform/build_config.bzl",
        content = """load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

def tf_proto_library(
        name,
        deps = [],
        protodeps = [],
        visibility = None,
        **kwargs):
    proto_library(
        name = name,
        visibility = visibility,
        deps = deps + protodeps,
        **kwargs,
    )
    cc_proto_library(
        name = name + "_cc",
        deps = [":{}".format(name)],
        visibility = visibility,
    )
""",
        executable = False,
    )

stub_tf_proto_library = repository_rule(_stub_tf_proto_library_impl)
