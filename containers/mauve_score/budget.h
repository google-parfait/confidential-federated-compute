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
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/common/intervals/interval.h"
#include "containers/common/time_budget/budget.pb.h"
#include "containers/common/time_budget/time_budget.h"

namespace confidential_federated_compute::mauve_score {

// Budget class serves as a migration layer between legacy per-key budget
// tracking and time-based budget tracking (TimeBudget) for the MAUVE score
// container.
//
// It maintains both schemes in parallel using the shared BudgetState proto:
// - Legacy per-key budgets: Tracked via per_key_budgets_, using KMS key IDs as
//   bucket identifiers.
// - Time-based budget: Tracked via time_budget_, using minute-granularity time
//   windows.
class Budget {
 public:
  // Creates a Budget with the given initial serialized BudgetState received
  // from KMS (empty string or nullopt for first run) and default access budget
  // from config constraints.
  static absl::StatusOr<Budget> Create(std::optional<std::string> initial_state,
                                       uint32_t default_budget);

  // This class is move-only.
  Budget(const Budget&) = delete;
  Budget& operator=(const Budget&) = delete;

  Budget(Budget&&) = default;
  Budget& operator=(Budget&&) = default;

  // Checks whether any budget remains for the specified bucket key.
  bool HasRemainingBudget(const std::string& key) const;

  // Checks whether any time-based budget remains for the specified time window.
  bool HasRemainingBudget(Interval<uint64_t> time_window);

  // Updates legacy per-key budgets for the given set of keys.
  // Returns an error if the pipeline has already transitioned to time-based
  // budget, or if any key in `keys` has exhausted its budget.
  absl::Status UpdatePerKeyBudget(const absl::flat_hash_set<std::string>& keys);

  // Updates time-based budget for the given time window.
  // Does not modify per_key_budgets_.
  absl::Status UpdateTimeBudget(Interval<uint64_t> time_window);

  // Returns the initial state received at construction.
  const std::optional<std::string>& GetInitialState() const {
    return initial_state_;
  }

  BudgetState Serialize() const;
  std::string SerializeAsString() const;

 private:
  Budget(std::optional<std::string> initial_state, uint32_t default_budget,
         absl::flat_hash_map<std::string, uint32_t> per_key_budgets,
         TimeBudget time_budget)
      : initial_state_(std::move(initial_state)),
        default_budget_(default_budget),
        per_key_budgets_(std::move(per_key_budgets)),
        time_budget_(std::move(time_budget)) {}

  std::optional<std::string> initial_state_;
  uint32_t default_budget_;
  absl::flat_hash_map<std::string, uint32_t> per_key_budgets_;
  TimeBudget time_budget_;
};

}  // namespace confidential_federated_compute::mauve_score

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MAUVE_SCORE_BUDGET_H_
