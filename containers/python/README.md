# Embedded Python

Containers whose entry point is a C++ binary that runs Python code in an
embedded CPython interpreter through pybind11 use two pieces:

*   `embedded_python_env` (`//containers/bazel:embedded_python_env.bzl`)
    collects the interpreter binary and standard library of the hermetic
    rules_python toolchain and the transitive closure of the given `py_library`
    targets as runfiles of the binary.
*   `EmbeddedInterpreter` (`:embedded_python`) initializes the interpreter from
    those runfiles in isolated mode.

The binary finds everything through the Bazel runfiles library. It works the
same way under `bazel run`, under `bazel test` and in a container image built
with `pkg_tar(include_runfiles = True)`, and it doesn't need PYTHONHOME,
PYTHONPATH or a Python installation in the base image. The interpreter,
libpython, and the standard library always come from the same rules_python
toolchain.

## Usage

```starlark
load("@confidential-federated-compute//containers/bazel:embedded_python_env.bzl", "embedded_python_env")

embedded_python_env(
    name = "py_env",
    deps = [":my_fn_py_lib"],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":py_env",
        "@confidential-federated-compute//containers/python:embedded_python",
        "@pybind11",
    ],
)

# Tests link an environment too; no `env` or `toolchains` attributes needed.
cc_test(
    name = "my_fn_test",
    srcs = ["my_fn_test.cc"],
    deps = [":py_env", ...],
)
```

```c++
// main.cc: once, before starting threads that use Python.
absl::StatusOr<std::unique_ptr<EmbeddedInterpreter>> interpreter =
    EmbeddedInterpreter::Create();
CHECK_OK(interpreter);
...
// On any thread, including this one:
pybind11::gil_scoped_acquire gil;
pybind11::module_::import("my_package.my_module").attr("run")();
```

`Create()` returns with the GIL released. It releases it with
`pybind11::gil_scoped_release`, which also creates pybind11's internal state on
the creating thread, so any thread can be the first to acquire the GIL.

## Rules

*   **Exactly one environment per binary.** A process has one interpreter, so
    all the Python code it runs, for example several Python fns in one
    container, must share one environment with one version of each package.
    Every binary and test that depends on `:embedded_python`, directly or
    through a library, links one `embedded_python_env`: none is a link error
    (undefined `GetEmbeddedPythonConfig()`), and so are two (duplicate
    definition). An environment in which the same distribution is reachable
    from two `sys.path` directories fails to build.
*   **One interpreter per process.** CPython and extension modules such as
    numpy can't be re-initialized, so `EmbeddedInterpreter::Create()` fails
    after the first call. Tests share one interpreter per test binary, for
    example through a gtest `Environment`.
*   **Package-qualified imports.** Prefer importing first-party code by its
    full package path over `imports = ["."]`, so that code from different fns
    can't collide on top-level module names.

## Python subprocesses

The interpreter binary is shipped (`bin/python3`, about 30 MB with the stripped
python-build-standalone build) and is `sys.executable`, so `subprocess` and
`multiprocessing` can start Python processes. `multiprocessing` passes
`sys.path` to its child processes, but a plain `subprocess` doesn't inherit the
`sys.path` of the embedded interpreter: it only sees the standard library.
