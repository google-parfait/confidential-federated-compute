// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GVISOR_PROCESS_RUNNER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GVISOR_PROCESS_RUNNER_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace confidential_federated_compute::gvisor {

struct ProcessResult {
  // Exit code if the process exited normally, or 128 + signal number if it was
  // killed by a signal.
  int exit_code = -1;
  bool timed_out = false;
  std::string stdout_data;
  std::string stderr_data;
  // Whether the process wrote more than `max_output_bytes` to the stream, so
  // that the data above is only its beginning.
  bool stdout_truncated = false;
  bool stderr_truncated = false;
};

// Runs `argv` (argv[0] must be an absolute path) as a child process, writes
// `stdin_data` to its stdin, and captures its stdout and stderr. Kills the
// process if it runs longer than `timeout`. Each captured stream is truncated
// to `max_output_bytes`.
absl::StatusOr<ProcessResult> RunProcess(const std::vector<std::string>& argv,
                                         absl::string_view stdin_data,
                                         absl::Duration timeout,
                                         size_t max_output_bytes = 1 << 20);

}  // namespace confidential_federated_compute::gvisor

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GVISOR_PROCESS_RUNNER_H_
