# Copyright 2026 Google LLC.
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
"""Hermetic Python environments for C++ binaries that embed Python.

`embedded_python_env` collects everything an embedded CPython interpreter needs
at runtime as runfiles of the C++ binary:

- the interpreter binary and the standard library of the hermetic rules_python
  toolchain, and
- the transitive closure of the given `py_library` targets (first-party code,
  pip packages and Python code from other Bazel repositories),

and generates the definition of
`confidential_federated_compute::python::GetEmbeddedPythonConfig()`, which
tells `EmbeddedInterpreter` (//containers/python:embedded_python) where to find
them. The interpreter finds everything through the runfiles library, so the
same binary works under `bazel run`, under `bazel test` and in a container
image built with `pkg_tar(include_runfiles = True)`, without PYTHONHOME,
PYTHONPATH or a Python installation in the base image.

Usage:

    embedded_python_env(
        name = "py_env",
        deps = [":my_fn_py_lib", ":my_other_fn_py_lib"],
    )

    cc_binary(
        name = "main",
        srcs = ["main.cc"],
        deps = [
            ":py_env",
            "@confidential-federated-compute//containers/python:embedded_python",
        ],
    )

Rules:

- Link exactly one `embedded_python_env` into each binary or test that depends
  on `//containers/python:embedded_python`, directly or through a library. A
  process has a single interpreter, so all Python code it runs must share one
  environment. Linking no environment fails with an undefined reference to
  `GetEmbeddedPythonConfig()`, linking two with a duplicate definition.
- Each distribution (pip package) may appear only once in an environment. The
  build fails if the same distribution is reachable from two `sys.path`
  directories, for example from two different pip hubs.
- Prefer package-qualified imports over `imports = ["."]` for first-party
  code, so that code from different Python fns can't collide on top-level
  module names.

The interpreter binary (`bin/python3`, about 30 MB with the stripped
python-build-standalone build) is shipped and used as `sys.executable`, so
`subprocess` and `multiprocessing` can start Python processes.
"""

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "PyInfo")

# Parts of the toolchain's lib/pythonX.Y that an embedded interpreter doesn't
# need. `site-packages` is excluded so that packages bundled with the toolchain
# (such as pip) never end up on sys.path.
_STDLIB_EXCLUDED_DIRS = [
    "/ensurepip/",
    "/idlelib/",
    "/lib2to3/",
    "/site-packages/",
    "/test/",
    "/tests/",
    "/tkinter/",
    "/turtledemo/",
]

def _rlocation(workspace_name, file):
    """Returns the runfiles path ("rlocation path") of a file."""
    short_path = file.short_path
    if short_path.startswith("../"):
        return short_path[3:]
    return workspace_name + "/" + short_path

def _normalize_distribution_name(name):
    # https://packaging.python.org/en/latest/specifications/name-normalization/
    return name.lower().replace("_", "-").replace(".", "-")

def _embedded_python_env_gen_impl(ctx):
    py_runtime = ctx.toolchains["@rules_python//python:toolchain_type"].py3_runtime
    py_cc_toolchain = ctx.toolchains["@rules_python//python/cc:toolchain_type"].py_cc_toolchain
    if not py_runtime or not py_runtime.interpreter:
        fail("embedded_python_env requires a hermetic (in-build) rules_python interpreter")

    # libpython and the headers (py_cc_toolchain) must come from the same Python
    # as the standard library (py_runtime).
    version = py_runtime.interpreter_version_info
    major_minor = "{}.{}".format(version.major, version.minor)
    full_version = "{}.{}".format(major_minor, version.micro)
    if py_cc_toolchain.python_version not in (full_version, major_minor):
        fail("libpython/headers (Python {}) and the standard library (Python {}) come from different toolchains".format(
            py_cc_toolchain.python_version,
            full_version,
        ))

    interpreter_rlocation = _rlocation(ctx.workspace_name, py_runtime.interpreter)

    # The toolchain's interpreter is <home>/bin/python3, so the Python home
    # (sys.prefix) is two directories up, and the standard library is in
    # <home>/lib/pythonX.Y.
    home_rlocation = interpreter_rlocation.rsplit("/", 2)[0]
    home_short_path = py_runtime.interpreter.short_path.rsplit("/", 2)[0]
    stdlib_prefix = "lib/python{}/".format(major_minor)

    runtime_files = []
    for file in py_runtime.files.to_list():
        relative_path = file.short_path[len(home_short_path) + 1:]
        if not relative_path.startswith(stdlib_prefix):
            continue
        if [d for d in _STDLIB_EXCLUDED_DIRS if d in "/" + relative_path]:
            continue
        runtime_files.append(file)
    runtime_files.append(py_runtime.interpreter)

    # sys.path: the main repository, the import roots of all dependencies
    # (pip packages, `imports` attributes), and the roots of other repositories
    # that contribute Python sources without `imports` (for example generated
    # protos), mirroring what rules_python does for py_binary.
    py_infos = [dep[PyInfo] for dep in ctx.attr.deps]
    imports = depset(transitive = [info.imports for info in py_infos]).to_list()
    repository_roots = {}
    for source in depset(transitive = [info.transitive_sources for info in py_infos]).to_list():
        if source.short_path.startswith("../"):
            repository_roots[source.short_path[3:].split("/")[0]] = True
    sys_path = [ctx.workspace_name] + imports + [
        root
        for root in sorted(repository_roots.keys())
        if not [i for i in imports if i == root or i.startswith(root + "/")]
    ]

    runfiles = ctx.runfiles(files = runtime_files)
    for dep in ctx.attr.deps:
        runfiles = runfiles.merge(dep[DefaultInfo].default_runfiles)

    # One process has one interpreter, which can only import one copy of each
    # distribution: the first one on sys.path would silently win.
    sys_path_prefixes = [path + "/" for path in sys_path]
    providers = {}
    for file in runfiles.files.to_list():
        path = _rlocation(ctx.workspace_name, file)
        if not path.endswith(".dist-info/METADATA"):
            continue
        for prefix in sys_path_prefixes:
            if not path.startswith(prefix):
                continue
            relative_path = path[len(prefix):]
            if relative_path.count("/") != 1:
                continue
            dist_info_dir = relative_path.split("/")[0][:-len(".dist-info")]
            distribution = _normalize_distribution_name(dist_info_dir.rsplit("-", 1)[0])
            providers.setdefault(distribution, {})[prefix[:-1]] = True
    duplicates = {
        distribution: sorted(paths.keys())
        for distribution, paths in providers.items()
        if len(paths) > 1
    }
    if duplicates:
        fail("Python distributions provided more than once in one embedded interpreter: {}".format(duplicates))

    config_cc = ctx.actions.declare_file(ctx.label.name + ".cc")
    ctx.actions.write(config_cc, "\n".join([
        "// Generated by embedded_python_env (%s). Do not edit." % ctx.label,
        "",
        '#include "containers/python/embedded_python.h"',
        "",
        "namespace confidential_federated_compute::python {",
        "namespace {",
        "",
        "constexpr const char* kSysPath[] = {",
    ] + ['    "{}",'.format(path) for path in sys_path] + [
        "    nullptr,",
        "};",
        "",
        "constexpr EmbeddedPythonConfig kConfig = {",
        '    /*home=*/"{}",'.format(home_rlocation),
        '    /*executable=*/"{}",'.format(interpreter_rlocation),
        "    /*sys_path=*/kSysPath,",
        "};",
        "",
        "}  // namespace",
        "",
        "const EmbeddedPythonConfig& GetEmbeddedPythonConfig() { return kConfig; }",
        "",
        "}  // namespace confidential_federated_compute::python",
        "",
    ]))
    return [DefaultInfo(files = depset([config_cc]), runfiles = runfiles)]

_embedded_python_env_gen = rule(
    implementation = _embedded_python_env_gen_impl,
    attrs = {
        "deps": attr.label_list(providers = [PyInfo]),
    },
    toolchains = [
        "@rules_python//python:toolchain_type",
        "@rules_python//python/cc:toolchain_type",
    ],
)

def embedded_python_env(name, deps, **kwargs):
    """A cc_library that provides the hermetic Python environment of a binary.

    Args:
      name: Name of the cc_library. Link exactly one per binary or test.
      deps: py_library targets whose code the embedded interpreter imports.
      **kwargs: Passed to the cc_library (e.g. testonly, visibility, tags).
    """
    _embedded_python_env_gen(
        name = name + "_gen",
        deps = deps,
        tags = kwargs.get("tags", []),
        testonly = kwargs.get("testonly", False),
        visibility = ["//visibility:private"],
    )
    cc_library(
        name = name,
        srcs = [":" + name + "_gen"],
        data = [":" + name + "_gen"],
        # Link the generated GetEmbeddedPythonConfig() into the executable
        # itself, also when tests are linked dynamically, so that linking two
        # environments into one binary always fails with a duplicate definition.
        alwayslink = True,
        linkstatic = True,
        deps = [Label("//containers/python:embedded_python")],
        **kwargs
    )
