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
  ~PrivateState() = default;

  // Construct a PrivateState instance based on the initial state, if provided.
  static absl::StatusOr<std::unique_ptr<PrivateState>> CreatePrivateState(
      std::optional<std::string> initial_state);

  // Whether recovery from a given blob is permitted.
  bool AllowRecovery(std::string recovery_blob_id,
                     const RecoveryInfo& recovery_info) const;

  // Whether any save recovery calls have been made in the current run or have
  // been committed to KMS in previous runs.
  bool HasPriorSaveRecovery() const;

  // Return the recovery blob ID from the last committed state.
  std::optional<std::string> GetCommittedRecoveryBlobId() const;

  // Update the recovery blob ID to use for the next release operation.
  void SetRecoveryBlobId(std::string blob_id);

  // Return the "initial state" for use in a release operation.
  std::optional<std::string> GetState() const;

  // Commit a new "update state" and return it for use in a release operation.
  std::string CommitNewState();

 protected:
  PrivateState(std::optional<std::string> initial_state,
               BudgetState internal_state)
      : committed_serialized_state_(std::move(initial_state)),
        internal_state_(std::move(internal_state)) {}

 private:
  // Each release operation produces a release token containing a "initial
  // state" that must match KMS's current stored state as well as an "update
  // state" that KMS should update its stored state to.
  // The "initial state" to use for the next release operation.
  std::optional<std::string> committed_serialized_state_;
  // The internal state from which the "update state" can be derived for a
  // release operation. Opon a release operation, this becomes the committed
  // state.
  BudgetState internal_state_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PRIVATE_STATE_H_