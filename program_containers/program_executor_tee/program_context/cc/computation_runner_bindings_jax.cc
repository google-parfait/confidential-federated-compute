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

#include <pybind11/pybind11.h>

#include "federated_language_jax/executor/xla_executor.h"
#include "program_executor_tee/program_context/cc/computation_runner_bindings.h"

namespace confidential_federated_compute::program_executor_tee {

PYBIND11_MODULE(computation_runner_bindings_jax, m) {
  m.doc() = "In-process ComputationRunner with XLA leaf executors, for JAX.";

  // The XLA executor does not support max_concurrent_computation_calls.
  DefineComputationRunner(m, [](int /*max_concurrent_computation_calls*/) {
    return federated_language_jax::CreateXLAExecutor();
  });
}

}  // namespace confidential_federated_compute::program_executor_tee
