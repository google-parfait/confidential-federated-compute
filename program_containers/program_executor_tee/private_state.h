// Copyright 2025 Google LLC.
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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PRIVATE_STATE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PRIVATE_STATE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "program_executor_tee/budget.pb.h"

namespace confidential_federated_compute::program_executor_tee {

// PrivateState is a wrapper class for the manipulating the program executor
// private state that is persisted via KMS.
class PrivateState {
 public:
  // Construct a PrivateState instance based on the initial state, if provided,
  // or the default max number of runs.
  static absl::StatusOr<std::unique_ptr<PrivateState>> CreatePrivateState(
      std::optional<std::string> initial_state, uint32_t default_max_num_runs);

  // Update the "initial state" to use for the next release operation.
  virtual void SetReleaseInitialState(std::string initial_state);

  // Return the "initial state" to use for the next release operation.
  virtual std::optional<std::string> GetReleaseInitialState() const;

  // Return the "update state" to use for the next release operation.
  virtual std::string GetReleaseUpdateState();

 protected:
  PrivateState(std::optional<std::string> initial_state,
               BudgetState next_update_state)
      : initial_serialized_state_(std::move(initial_state)),
        next_update_state_(next_update_state) {}

 private:
  // Each release operation produces a release token containing a "initial
  // state" that must match KMS's current stored state as well as an "update
  // state" that KMS should update its stored state to.
  // The "initial state" to use for the next release operation.
  std::optional<std::string> initial_serialized_state_;
  // The "update state" to use for the next release operation. This field is not
  // initialized until InitializeState is called.
  BudgetState next_update_state_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PRIVATE_STATE_H_