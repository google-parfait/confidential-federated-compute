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
    dep_imports = [dep[PyInfo].imports for dep in ctx.attr.py_deps]
    for path in depset(transitive = dep_imports).to_list():
        python_path += runfiles_dir + path + ":"
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
