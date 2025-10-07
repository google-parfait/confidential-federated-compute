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

"""Define rule enabling python execution with external dependencies."""

def _trusted_runtime_impl(ctx):
    """Implementation of trusted_py_runtime rule."""
    py3_runtime = ctx.toolchains["@bazel_tools//tools/python:toolchain_type"].py3_runtime

    # Collect runfiles from the py3_runtime and the provided py deps.
    dep_runfiles = [dep[DefaultInfo].default_runfiles for dep in ctx.attr.py_deps]
    runfiles = ctx.runfiles(files = py3_runtime.files.to_list()).merge_all(dep_runfiles)

    # Construct the PYTHONPATH from the provided py deps and source repos.
    python_path = ""
    dep_imports = [dep[PyInfo].imports for dep in ctx.attr.py_deps]
    for path in depset(transitive = dep_imports).to_list():
        python_path += "external/" + path + ":"
    for source_repo in ctx.attr.source_repo_names:
        python_path += "external/" + source_repo + ":"

    return [
        DefaultInfo(runfiles = runfiles),
        # A collection of environment variables that should be set using the
        # `env` attribute in a cc_binary or cc_test rule that is depending on
        # a trusted_py_runtime target via the `toolchains` attribute.
        platform_common.TemplateVariableInfo({
            "PYTHONPATH": python_path,
            "PYTHONHOME": str(py3_runtime.interpreter.dirname.rstrip("bin")),
        }),
    ]

trusted_py_runtime = rule(
    implementation = _trusted_runtime_impl,
    attrs = {
        "py_deps": attr.label_list(
            providers = [PyInfo],
            doc = """Python dependencies, including custom py_library targets
                     and targets installed via pip.""",
        ),
        "source_repo_names": attr.string_list(
            doc = """Names of source repos from the WORKSPACE file that are
                     needed at runtime.""",
        ),
    },
    toolchains = [
        str(Label("@bazel_tools//tools/python:toolchain_type")),
    ],
    doc = """A rule to collect python runfiles and environment variables needed
             to execute python code with external depenencies from C++. A
             target of this type can be provided to the `data` attribute of a
             cc_library rule to provide runfiles. A target of this type can be
             provided to the `toolchains` attribute of a cc_binary or cc_test
             rule to provide the TemplateVariableInfo that contains environment
             variables that should be set on the `env` attribute of the same
             cc_binary or cc_test rule.""",
)
