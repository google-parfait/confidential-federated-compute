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

#include "program_executor_tee/private_state.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"

namespace confidential_federated_compute::program_executor_tee {

absl::StatusOr<std::unique_ptr<PrivateState>> PrivateState::CreatePrivateState(
    std::optional<std::string> initial_state) {
  BudgetState internal_state;
  if (initial_state.has_value()) {
    if (!internal_state.ParseFromString(initial_state.value())) {
      return absl::InvalidArgumentError(
          "Failed to parse initial state into BudgetState.");
    }
    // If there is an initial state but it does not contain recovery
    // information, it has already been run once before without support for
    // recovery.
    if (!internal_state.has_recovery_blob_id()) {
      return absl::FailedPreconditionError(
          "Programs may recover, but cannot run from scratch multiple times.");
    }
  }
  return std::unique_ptr<PrivateState>(
      new PrivateState(std::move(initial_state), std::move(internal_state)));
}

bool PrivateState::AllowRecovery(std::string recovery_blob_id,
                                 const RecoveryInfo& recovery_info) const {
  // Recovery from a specific blob is allowed in the cases described below. Note
  // that the committed blob id in the blob's RecoveryInfo message represents
  // the recovery blob id in the BudgetState tracked by KMS at the time the
  // RecoveryInfo was generated. Permitted cases:
  // - The BudgetState tracked by KMS does not have a recovery blob id and the
  // RecoveryInfo does not have a committed blob id. This case represents
  // recovering from RecoveryInfo before any unencrypted value releases have
  // triggered commits to KMS.
  // - The recovery blob id in the BudgetState tracked by KMS matches this
  // recovery blob's id. This case represents recovering from the very last
  // RecoveryInfo message that was produced by the TEE.
  // - The recovery blob id in the BudgetState tracked by KMS matches the
  // committed blob id in the blob's RecoveryInfo message. This case represents
  // recovering from a RecoveryInfo message that is earlier than the last
  // RecoveryInfo message that was produced by the TEE but later than the
  // RecoveryInfo message released alongside the latest unencrypted results
  // (which would have been the last time an update to KMS was triggered).
  // Recovery is safe because we will not be able to go back in time to release
  // new versions of previously released unencrypted values.
  if (!internal_state_.has_recovery_blob_id()) {
    return !recovery_info.has_committed_blob_id();
  }
  return internal_state_.recovery_blob_id() == recovery_blob_id ||
         (recovery_info.has_committed_blob_id() &&
          internal_state_.recovery_blob_id() ==
              recovery_info.committed_blob_id());
}

bool PrivateState::HasPriorSaveRecovery() const {
  return internal_state_.has_recovery_blob_id();
}

std::optional<std::string> PrivateState::GetCommittedRecoveryBlobId() const {
  if (!committed_serialized_state_.has_value()) {
    return std::nullopt;
  }
  BudgetState committed_state;
  if (!committed_state.ParseFromString(committed_serialized_state_.value())) {
    return std::nullopt;
  }
  if (!committed_state.has_recovery_blob_id()) {
    return std::nullopt;
  }
  return committed_state.recovery_blob_id();
}

void PrivateState::SetRecoveryBlobId(std::string blob_id) {
  internal_state_.set_recovery_blob_id(std::move(blob_id));
}

std::optional<std::string> PrivateState::GetState() const {
  return committed_serialized_state_;
}

std::string PrivateState::CommitNewState() {
  // The number of runs was already adjusted when this object was constructed,
  // and only the counter field should continue to change with each release.
  internal_state_.set_counter(internal_state_.counter() + 1);
  committed_serialized_state_ = internal_state_.SerializeAsString();
  return *committed_serialized_state_;
}

}  // namespace confidential_federated_compute::program_executor_tee