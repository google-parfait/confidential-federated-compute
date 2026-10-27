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

#include "program_executor_tee/program_context/cc/computation_runner_bindings.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

namespace confidential_federated_compute::program_executor_tee {

PYBIND11_MODULE(computation_runner_bindings_tensorflow, m) {
  m.doc() = "In-process ComputationRunner with TensorFlow leaf executors.";

  DefineComputationRunner(m, [](int max_concurrent_computation_calls) {
    return tensorflow_federated::CreateTensorFlowExecutor(
        max_concurrent_computation_calls,
        /*synchronous_value_creation=*/true);
  });
}

}  // namespace confidential_federated_compute::program_executor_tee
