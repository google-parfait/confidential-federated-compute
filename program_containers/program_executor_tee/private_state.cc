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
    std::optional<std::string> initial_state, uint32_t default_max_num_runs) {
  BudgetState next_update_state;
  if (initial_state.has_value()) {
    if (!next_update_state.ParseFromString(*initial_state)) {
      return absl::InternalError("Failed to parse initial budget state.");
    }
  } else {
    next_update_state.set_num_runs_remaining(default_max_num_runs);
  }
  // Decrease the number of runs remaining to account for the program that will
  // run.
  next_update_state.set_num_runs_remaining(
      next_update_state.num_runs_remaining() - 1);
  return std::unique_ptr<PrivateState>(
      new PrivateState(std::move(initial_state), std::move(next_update_state)));
}

void PrivateState::SetReleaseInitialState(std::string initial_state) {
  initial_serialized_state_ = initial_state;
}

std::optional<std::string> PrivateState::GetReleaseInitialState() const {
  return initial_serialized_state_;
}

std::string PrivateState::GetReleaseUpdateState() {
  // The number of runs was already adjusted when this object was constructed,
  // and only the counter field should continue to change with each release.
  next_update_state_.set_counter(next_update_state_.counter() + 1);
  return next_update_state_.SerializeAsString();
}

}  // namespace confidential_federated_compute::program_executor_tee