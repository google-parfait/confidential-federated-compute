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

"""Rule for converting protocol buffers from text to binary format."""

load("@rules_proto//proto:defs.bzl", "ProtoInfo")

def _proto_data_impl(ctx):
    direct_proto_deps = [
        dep[ProtoInfo].direct_descriptor_set
        for dep in ctx.attr.proto_deps
    ]
    transitive_proto_deps = [
        dep[ProtoInfo].transitive_descriptor_sets
        for dep in ctx.attr.proto_deps
    ]
    proto_deps = depset(direct_proto_deps, transitive = transitive_proto_deps)

    ctx.actions.run_shell(
        inputs = [ctx.file.src] + proto_deps.to_list(),
        outputs = [ctx.outputs.out],
        tools = [ctx.executable._protoc],
        command = "{} --encode={} --deterministic_output --descriptor_set_in={} < {} > {}".format(
            ctx.executable._protoc.path,
            ctx.attr.proto_name,
            ":".join([file.path for file in proto_deps.to_list()]),
            ctx.file.src.path,
            ctx.outputs.out.path,
        ),
    )

_proto_data = rule(
    implementation = _proto_data_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "out": attr.output(mandatory = True),
        "proto_name": attr.string(mandatory = True),
        "proto_deps": attr.label_list(
            providers = [ProtoInfo],
            mandatory = True,
        ),
        "_protoc": attr.label(
            default = Label("@com_google_protobuf//:protoc"),
            executable = True,
            cfg = "exec",
        ),
    },
)

def proto_data(
        name,
        src,
        proto_name,
        proto_deps,
        out = None,
        **kwargs):
    """Converts a text-format protocol buffer to binary format.

    Args:
      name: The name of the rule.
      src: The text-format protocol buffer file.
      proto_name: The fully-qualified name of the protocol buffer message.
      proto_deps: The list of proto_library targets that message definition.
      out: The name of the output file. Defaults to "{name}.binarypb".
      **kwargs: Additional common arguments.
    """
    _proto_data(
        name = name,
        src = src,
        out = out or "{}.binarypb".format(name),
        proto_name = proto_name,
        proto_deps = proto_deps,
        **kwargs
    )
