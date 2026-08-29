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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MAUVE_SCORE_BUDGET_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MAUVE_SCORE_BUDGET_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace confidential_federated_compute::mauve_score {

// Manages budget for the MAUVE container.
//
// The budget tracks how many times the MAUVE pipeline can process data.
// On each run, the remaining budget is decremented by 1.
//
// Usage:
//   ASSIGN_OR_RETURN(Budget budget, Budget.Create(initial_state,
//   access_budget_times));
//   // ... process data ...
//   budget.DecrementBudget();
//   ...
//   context.EmitReleasable(..., budget.GetInitialState(),
//   budget.SerializeAsString());
class Budget {
 public:
  // Creates a Budget with the given initial state received from KMS
  // (empty string for first run) and access budget from config constraints.
  static absl::StatusOr<Budget> Create(std::optional<std::string> initial_state,
                                       uint32_t access_budget_times);

  void DecrementBudget();

  // Returns the initial state received at construction.
  const std::optional<std::string>& GetInitialState() const {
    return initial_state_;
  }
  std::string SerializeAsString() const;

 private:
  Budget(std::optional<std::string> initial_state, uint32_t remaining_budget)
      : initial_state_(std::move(initial_state)),
        remaining_budget_(remaining_budget) {};

  std::optional<std::string> initial_state_;
  uint32_t remaining_budget_;
};

}  // namespace confidential_federated_compute::mauve_score

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MAUVE_SCORE_BUDGET_H_
