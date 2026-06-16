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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MAUVE_SCORE_PY_MAUVE_DELEGATE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MAUVE_SCORE_PY_MAUVE_DELEGATE_H_

#include <pybind11/embed.h>

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/protos/confidentialcompute/mauve_score_config.pb.h"

namespace confidential_federated_compute::mauve_score {

// Calls compute_mauve_score_lib.compute_mauve_and_serialize_result() via
// pybind11. Takes real and synthetic embedding matrices (as vectors of
// vectors), computes MAUVE entirely in Python, and returns the parsed
// MauveScoreResult proto.
absl::StatusOr<fcp::confidentialcompute::MauveScoreResult>
ComputeMauveViaPython(const std::vector<std::vector<float>>& real_embeddings,
                      const std::vector<std::vector<float>>& synth_embeddings);

// Manages the Python interpreter lifecycle.
// Must be created on the main thread before any gRPC calls.
class PyRuntimeManager {
 public:
  PyRuntimeManager();
  ~PyRuntimeManager();

  // Import the compute_mauve_score_lib Python module.
  // Must be called before ComputeMauveViaPython().
  absl::Status ImportLib();

 private:
  PyThreadState* tstate_;
};

}  // namespace confidential_federated_compute::mauve_score

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MAUVE_SCORE_PY_MAUVE_DELEGATE_H_
