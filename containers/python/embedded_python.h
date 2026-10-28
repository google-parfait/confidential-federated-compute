// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PYTHON_EMBEDDED_PYTHON_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PYTHON_EMBEDDED_PYTHON_H_

#include <pybind11/embed.h>

#include <memory>
#include <optional>

#include "absl/status/statusor.h"

namespace confidential_federated_compute::python {

// Runfiles paths ("rlocation paths") of the hermetic Python environment of a
// binary. See //containers/bazel:embedded_python_env.bzl.
struct EmbeddedPythonConfig {
  // The root of the Python installation (PyConfig.home), which contains the
  // standard library in lib/pythonX.Y.
  const char* home;
  // The Python interpreter, used as sys.executable.
  const char* executable;
  // nullptr-terminated list of directories that are added to sys.path with
  // site.addsitedir(), which also processes .pth files.
  const char* const* sys_path;
};

// Defined by the embedded_python_env target linked into the binary. Linking no
// such target fails with an undefined reference, linking two fails with a
// duplicate definition.
const EmbeddedPythonConfig& GetEmbeddedPythonConfig();

// The embedded Python interpreter of the process, initialized from the hermetic
// environment shipped in the binary's runfiles. It doesn't depend on
// PYTHONHOME, PYTHONPATH or a Python installation on the machine: the
// interpreter runs in isolated mode and ignores PYTHON* environment variables.
//
// Create it once, before other threads use Python, and destroy it on the same
// thread. That thread becomes Python's main thread (for example, only it can
// install signal handlers). When Create() returns, the GIL is released: every
// thread, including the creating one, uses pybind11::gil_scoped_acquire to call
// into Python.
class EmbeddedInterpreter {
 public:
  // Initializes the interpreter. Fails if this process has already initialized
  // a Python interpreter: CPython and extension modules such as numpy don't
  // support initializing one again.
  static absl::StatusOr<std::unique_ptr<EmbeddedInterpreter>> Create();

  EmbeddedInterpreter(const EmbeddedInterpreter&) = delete;
  EmbeddedInterpreter& operator=(const EmbeddedInterpreter&) = delete;

 private:
  EmbeddedInterpreter() = default;

  // Destroyed in reverse order: gil_release_ reacquires the GIL, then
  // interpreter_ finalizes the interpreter.
  std::optional<pybind11::scoped_interpreter> interpreter_;
  // Uses pybind11::gil_scoped_release rather than PyEval_SaveThread(): it also
  // creates pybind11's internal state on this thread. Otherwise the first
  // pybind11::gil_scoped_acquire on another thread creates that state, keeps a
  // pointer to a thread state that Python deletes right away, and crashes the
  // next time that thread acquires the GIL (a segfault with Python 3.13).
  std::optional<pybind11::gil_scoped_release> gil_release_;
};

}  // namespace confidential_federated_compute::python

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PYTHON_EMBEDDED_PYTHON_H_
