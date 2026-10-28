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

// Initializes the embedded interpreter and prints where it found Python and the
// test packages, for embedded_python_smoke_test.

#include <pybind11/embed.h>

#include <iostream>

#include "containers/python/embedded_python.h"

int main() {
  auto interpreter =
      confidential_federated_compute::python::EmbeddedInterpreter::Create();
  if (!interpreter.ok()) {
    std::cerr << interpreter.status() << std::endl;
    return 1;
  }
  pybind11::gil_scoped_acquire gil;
  try {
    // Python flushes its stdout when the interpreter is finalized.
    pybind11::exec(R"py(
import os, subprocess, sys
import pip_style_package
from containers.python.testdata.greeting import greeting
print("prefix=" + sys.prefix)
print("os=" + os.__file__)
print("greeting=" + greeting.greet())
print("pip_style_package=" + pip_style_package.describe())
child = subprocess.run([sys.executable, "-I", "-c", "print('ok')"],
                       capture_output=True, text=True, check=True)
print("subprocess=" + child.stdout.strip())
)py");
  } catch (const pybind11::error_already_set& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
