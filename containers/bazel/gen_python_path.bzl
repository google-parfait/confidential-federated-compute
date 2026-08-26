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
"""Define rule for generating PYTHONPATH"""

def _gen_python_path_impl(ctx):
    """Generate the PYTHONPATH for python dependencies used by the main cc binary.

    This rule is designed to work with a cc_binary calling python library via pybind.
    The Python dependencies used by the python library are expected to be introduced
    via Bzlmod. The output is a text file which is used as ENV file for an oci_image.
    In addition to Bzlmod pip packages (whose paths come from PyInfo.imports),
    this rule also discovers repo-root paths for normal Bazel deps (e.g.
    @federated_compute//fcp/...) by inspecting transitive source short_paths.
    """
    output_file = ctx.actions.declare_file(ctx.attr.name)
    main = ctx.attr.cc_binary_name

    # The runfiles directory of main binary contains all the python dependencies
    # that are used by the embeded python library.
    runfiles_dir = "/{}.runfiles/".format(main)

    # The .py files for the embeded python library is under `/main.runfiles/_main/`
    py_script_dir = "{}_{}/".format(runfiles_dir, main)

    # Construct the PYTHONPATH from the provided py deps and source repos.
    python_path = "PYTHONPATH={}:".format(py_script_dir)

    # 1) Add paths from PyInfo.imports (covers pip/Bzlmod packages).
    dep_imports = [dep[PyInfo].imports for dep in ctx.attr.py_deps]
    for path in depset(transitive = dep_imports).to_list():
        python_path += runfiles_dir + path + ":"

    # 2) Add repo-root paths from transitive sources (covers normal Bazel deps
    #    like @federated_compute//fcp/...).
    #
    #    For external repos, File.short_path has the form:
    #        ../<canonical_repo_name>/package/file.py
    #    The runfiles-relative path is short_path with "../" stripped:
    #        <canonical_repo_name>/package/file.py
    #    The canonical_repo_name includes any Bzlmod suffix (e.g. "~1.0.0",
    #    "+1.0.0"), and the runfiles tree uses the same name, so this is safe.
    all_sources = [dep[PyInfo].transitive_sources for dep in ctx.attr.py_deps]
    repo_roots = {}
    for src in depset(transitive = all_sources).to_list():
        short = src.short_path
        if short.startswith("../"):
            runfiles_rel = short[3:]  # strip "../" to get runfiles-relative path
            repo_root = runfiles_rel.split("/")[0]  # first component = repo dir
            if repo_root not in repo_roots:
                repo_roots[repo_root] = True
                python_path += runfiles_dir + repo_root + ":"
    python_path += "$PYTHONPATH"
    ctx.actions.write(
        output = output_file,
        content = python_path,
    )
    return [DefaultInfo(files = depset([output_file]))]

gen_python_path = rule(
    implementation = _gen_python_path_impl,
    attrs = {
        "cc_binary_name": attr.string(
            doc = "The name for the cc binary",
            mandatory = True,
        ),
        "py_deps": attr.label_list(
            providers = [PyInfo],
            doc = """Custom py_library targets embeded in the cc_binary""",
        ),
    },
)
