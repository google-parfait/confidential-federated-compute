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
#include "budget.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mauve_budget_state.pb.h"

namespace confidential_federated_compute::mauve_score {

absl::StatusOr<Budget> Budget::Create(std::optional<std::string> initial_state,
                                      uint32_t access_budget_times) {
  uint32_t remaining_budget = access_budget_times;
  if (initial_state.has_value()) {
    MauveBudgetState state;
    if (!state.ParseFromString(*initial_state)) {
      return absl::InvalidArgumentError(
          "Failed to parse pipeline state as MauveBudgetState.");
    }
    if (state.has_remaining_budget()) {
      remaining_budget = state.remaining_budget();
    } else {
      remaining_budget = 0;
    }
  }

  if (remaining_budget <= 0) {
    return absl::ResourceExhaustedError(
        "Budget exhausted: remaining budget must be greater than zero.");
  }

  return Budget(std::move(initial_state), remaining_budget);
}

void Budget::DecrementBudget() { remaining_budget_--; }

std::string Budget::SerializeAsString() const {
  MauveBudgetState state;
  state.set_remaining_budget(remaining_budget_);
  return state.SerializeAsString();
}

}  // namespace confidential_federated_compute::mauve_score
