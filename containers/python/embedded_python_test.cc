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

#include "containers/python/embedded_python.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::python {
namespace {

namespace py = ::pybind11;
using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::EndsWith;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::StartsWith;

constexpr char kIgnoredPythonPath[] = "/nonexistent/pythonpath";

// Set by the test environment: whether a thread other than the one that
// created the interpreter could acquire the GIL first.
bool worker_thread_acquired_gil = false;

// A process can only initialize an interpreter once, so all tests share one.
class EmbeddedInterpreterEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    // The interpreter runs in isolated mode and must ignore these.
    setenv("PYTHONPATH", kIgnoredPythonPath, /*overwrite=*/1);
    setenv("PYTHONHOME", "/nonexistent/pythonhome", /*overwrite=*/1);
    absl::StatusOr<std::unique_ptr<EmbeddedInterpreter>> interpreter =
        EmbeddedInterpreter::Create();
    ASSERT_THAT(interpreter, IsOk());
    interpreter_ = *std::move(interpreter);

    // The first GIL acquisition after Create() happens on another thread, as in
    // a gRPC server that imports Python code lazily.
    std::thread worker([] {
      py::gil_scoped_acquire gil;
      py::module_::import("json");
      worker_thread_acquired_gil = true;
    });
    worker.join();
  }

  void TearDown() override { interpreter_.reset(); }

 private:
  std::unique_ptr<EmbeddedInterpreter> interpreter_;
};

const ::testing::Environment* const kEnvironment =
    ::testing::AddGlobalTestEnvironment(new EmbeddedInterpreterEnvironment);

std::string ToString(const py::handle& object) {
  return py::str(object).cast<std::string>();
}

py::module_ Sys() { return py::module_::import("sys"); }

TEST(EmbeddedInterpreterTest, WorkerThreadCanAcquireGilFirst) {
  EXPECT_TRUE(worker_thread_acquired_gil);
}

TEST(EmbeddedInterpreterTest, UsesStandardLibraryFromRunfiles) {
  py::gil_scoped_acquire gil;
  std::string prefix = ToString(Sys().attr("prefix"));
  EXPECT_THAT(prefix, HasSubstr(".runfiles/"));
  EXPECT_THAT(ToString(py::module_::import("os").attr("__file__")),
              StartsWith(prefix + "/lib/python"));
}

TEST(EmbeddedInterpreterTest, StandardLibraryMatchesLibpython) {
  py::gil_scoped_acquire gil;
  // PY_MAJOR_VERSION and PY_MINOR_VERSION come from the headers of the
  // libpython this test links; os.py comes from the standard library.
  EXPECT_THAT(ToString(py::module_::import("os").attr("__file__")),
              EndsWith(absl::StrCat("/lib/python", PY_MAJOR_VERSION, ".",
                                    PY_MINOR_VERSION, "/os.py")));
}

TEST(EmbeddedInterpreterTest, IgnoresPythonEnvironmentVariables) {
  py::gil_scoped_acquire gil;
  py::module_ sys = Sys();
  EXPECT_EQ(sys.attr("flags").attr("isolated").cast<int>(), 1);
  EXPECT_EQ(sys.attr("flags").attr("ignore_environment").cast<int>(), 1);
  EXPECT_THAT(sys.attr("path").cast<std::vector<std::string>>(),
              Not(Contains(kIgnoredPythonPath)));
}

TEST(EmbeddedInterpreterTest, ExecutableIsHermeticInterpreter) {
  py::gil_scoped_acquire gil;
  std::string executable = ToString(Sys().attr("executable"));
  EXPECT_THAT(executable, StartsWith(ToString(Sys().attr("prefix")) + "/"));
  EXPECT_THAT(executable, EndsWith("/bin/python3"));
}

TEST(EmbeddedInterpreterTest, ImportsFirstPartyCode) {
  py::gil_scoped_acquire gil;
  py::module_ greeting =
      py::module_::import("containers.python.testdata.greeting.greeting");
  EXPECT_EQ(ToString(greeting.attr("greet")()), "hello from embedded python");
}

// Pip packages are added to sys.path through their `imports` directory.
TEST(EmbeddedInterpreterTest, ImportsPipStylePackages) {
  py::gil_scoped_acquire gil;
  EXPECT_EQ(
      ToString(py::module_::import("pip_style_package").attr("describe")()),
      "imported from site-packages");
}

TEST(EmbeddedInterpreterTest, SubprocessRunsHermeticPython) {
  py::gil_scoped_acquire gil;
  py::dict scope;
  py::exec(R"py(
import os, subprocess, sys
child = subprocess.run(
    [sys.executable, "-I", "-c", "import sys; print(sys.prefix)"],
    capture_output=True, text=True, check=True)
# The child may see the prefix with symlinks resolved.
same_prefix = (os.path.realpath(child.stdout.strip())
               == os.path.realpath(sys.prefix))
)py",
           scope);
  EXPECT_TRUE(scope["same_prefix"].cast<bool>());
}

TEST(EmbeddedInterpreterTest, MultiprocessingSpawnWorks) {
  py::gil_scoped_acquire gil;
  py::dict scope;
  // The child process imports greet() through the sys.path that
  // multiprocessing passes to it.
  py::exec(R"py(
import multiprocessing
from containers.python.testdata.greeting import greeting
with multiprocessing.get_context("spawn").Pool(1) as pool:
  result = pool.apply(greeting.greet)
)py",
           scope);
  EXPECT_EQ(ToString(scope["result"]), "hello from embedded python");
}

TEST(EmbeddedInterpreterTest, CreateFailsIfAlreadyInitialized) {
  EXPECT_THAT(EmbeddedInterpreter::Create(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace confidential_federated_compute::python
