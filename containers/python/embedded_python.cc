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
#include <unistd.h>

#include <atomic>
#include <climits>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "rules_cc/cc/runfiles/runfiles.h"

namespace confidential_federated_compute::python {
namespace {

using ::rules_cc::cc::runfiles::Runfiles;

// Whether this process has initialized an interpreter. Py_IsInitialized() is
// false again after finalization, but initializing a second interpreter isn't
// supported.
std::atomic<bool> interpreter_initialized = false;

absl::StatusOr<std::unique_ptr<Runfiles>> CreateRunfiles() {
  // Locate the runfiles from the binary's own path rather than argv[0], which
  // may be a bare name (PATH lookup) or a symlink in another directory.
  // RUNFILES_DIR and RUNFILES_MANIFEST_FILE, which `bazel test` sets, still
  // take precedence.
  char executable[PATH_MAX];
  ssize_t length =
      readlink("/proc/self/exe", executable, sizeof(executable) - 1);
  if (length <= 0) {
    return absl::InternalError("Failed to read /proc/self/exe");
  }
  executable[length] = '\0';
  std::string error;
  std::unique_ptr<Runfiles> runfiles(
      Runfiles::Create(executable, BAZEL_CURRENT_REPOSITORY, &error));
  if (runfiles == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Failed to find the runfiles of ", executable, ": ", error));
  }
  return runfiles;
}

// Returns the absolute path of a runfile. Rlocation() doesn't check that the
// path exists.
absl::StatusOr<std::string> Resolve(const Runfiles& runfiles,
                                    const char* rlocation_path) {
  std::string path = runfiles.Rlocation(rlocation_path);
  if (path.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Runfile not found: ", rlocation_path));
  }
  std::error_code error;
  std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    return absl::InternalError(
        absl::StrCat("Failed to make ", path, " absolute: ", error.message()));
  }
  return absolute.lexically_normal().string();
}

absl::Status SetConfigString(PyConfig& config, wchar_t** field,
                             const std::string& value) {
  PyStatus status = PyConfig_SetBytesString(&config, field, value.c_str());
  if (PyStatus_Exception(status)) {
    return absl::InternalError(
        absl::StrCat("PyConfig_SetBytesString failed: ",
                     status.err_msg != nullptr ? status.err_msg : "unknown"));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<EmbeddedInterpreter>>
EmbeddedInterpreter::Create() {
  if (Py_IsInitialized() || interpreter_initialized) {
    return absl::FailedPreconditionError(
        "A Python interpreter has already been initialized in this process");
  }
  const EmbeddedPythonConfig& env = GetEmbeddedPythonConfig();
  absl::StatusOr<std::unique_ptr<Runfiles>> runfiles = CreateRunfiles();
  if (!runfiles.ok()) return runfiles.status();
  absl::StatusOr<std::string> home = Resolve(**runfiles, env.home);
  if (!home.ok()) return home.status();
  if (!std::filesystem::is_directory(*home)) {
    return absl::NotFoundError(
        absl::StrCat("The Python home ", *home, " doesn't exist"));
  }
  absl::StatusOr<std::string> executable = Resolve(**runfiles, env.executable);
  if (!executable.ok()) return executable.status();

  PyConfig config;
  // Isolated mode: ignore PYTHON* environment variables and the user's
  // site-packages directory, and leave signal handling to the C++ program.
  PyConfig_InitIsolatedConfig(&config);
  // The location of the standard library. This also disables the search for a
  // pyvenv.cfg file and for an interpreter on PATH.
  absl::Status status = SetConfigString(config, &config.home, *home);
  if (status.ok()) {
    status = SetConfigString(config, &config.executable, *executable);
  }
  if (!status.ok()) {
    PyConfig_Clear(&config);
    return status;
  }
  // The container's root file system may be read-only.
  config.write_bytecode = 0;

  auto interpreter = absl::WrapUnique(new EmbeddedInterpreter());
  interpreter_initialized = true;
  try {
    // Clears `config`.
    interpreter->interpreter_.emplace(&config, /*argc=*/0, /*argv=*/nullptr,
                                      /*add_program_dir_to_path=*/false);
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("Failed to initialize Python: ", e.what()));
  }
  try {
    pybind11::module_ site = pybind11::module_::import("site");
    for (const char* const* entry = env.sys_path; *entry != nullptr; ++entry) {
      absl::StatusOr<std::string> directory = Resolve(**runfiles, *entry);
      if (!directory.ok()) return directory.status();
      site.attr("addsitedir")(*directory);
    }
  } catch (const pybind11::error_already_set& e) {
    return absl::InternalError(
        absl::StrCat("Failed to set up sys.path: ", e.what()));
  }
  LOG(INFO) << "Initialized embedded Python " << Py_GetVersion()
            << " with home " << *home;
  interpreter->gil_release_.emplace();
  return interpreter;
}

}  // namespace confidential_federated_compute::python
