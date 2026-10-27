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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_RUNNER_BINDINGS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_RUNNER_BINDINGS_H_

#include <pybind11/pybind11.h>

#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace confidential_federated_compute::program_executor_tee {

// Returns a new leaf executor. `max_concurrent_computation_calls` is the number
// of calls that the executor can handle in parallel. Non-positive values
// indicate no maximum.
using LeafExecutorFactory = std::function<
    absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>(
        int max_concurrent_computation_calls)>;

// Adds the `ComputationRunner` Python class to `m`. Each leaf executor backend
// has its own bindings module, which calls this with a factory for the
// backend's leaf executors.
void DefineComputationRunner(pybind11::module_& m,
                             LeafExecutorFactory create_leaf_executor);

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_RUNNER_BINDINGS_H_
