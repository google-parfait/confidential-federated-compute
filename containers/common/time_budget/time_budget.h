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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_TIME_BUDGET_TIME_BUDGET_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_TIME_BUDGET_TIME_BUDGET_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "containers/common/intervals/interval.h"
#include "containers/common/time_budget/budget.pb.h"
#include "containers/common/time_budget/budget_interval_map.h"

namespace confidential_federated_compute {

// Time-based budget tracking implementation using minute-level granularity
// buckets.
class TimeBudget {
 public:
  // Creates an instance of TimeBudget with the specified default budget value.
  // If the default budget isn't specified (std::nullopt), it is considered to
  // be infinite/unlimited.
  explicit TimeBudget(std::optional<uint32_t> default_budget)
      : default_budget_(default_budget) {
    if (default_budget_.has_value() && default_budget_.value() > 0) {
      budget_map_.emplace(default_budget_.value());
    }
  }

  // This class is move-only.
  TimeBudget(const TimeBudget&) = delete;
  TimeBudget& operator=(const TimeBudget&) = delete;

  TimeBudget(TimeBudget&&) = default;
  TimeBudget& operator=(TimeBudget&&) = default;

  // Deserializes TimeBudget from the BudgetState::TimeBudgetState message.
  absl::Status Parse(const BudgetState::TimeBudgetState& state);

  // Deserializes TimeBudget from a string.
  absl::Status Parse(const std::string& data);

  // Serializes the current budget to a BudgetState::TimeBudgetState message.
  BudgetState::TimeBudgetState Serialize() const;

  // Serializes the current budget to a string.
  std::string SerializeAsString() const;

  // Checks whether any budget remains for the specified time window.
  bool HasRemainingBudget(Interval<uint64_t> time_window);

  // Checks whether the default budget is unlimited.
  bool HasUnlimitedBudget() const { return !default_budget_.has_value(); }

  // Update the budget for the specified time window.
  absl::Status UpdateBudget(Interval<uint64_t> time_window);

  // Gets the current anchor time (in UTC minutes since epoch).
  // This is the start time of the oldest stored interval.
  std::optional<uint64_t> anchor_time() const {
    if (budget_map_.has_value() && !budget_map_->empty()) {
      return budget_map_->begin()->first.start();
    }
    return std::nullopt;
  }

 private:
  std::optional<uint32_t> default_budget_;
  std::optional<BudgetIntervalMap> budget_map_;
};

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_TIME_BUDGET_TIME_BUDGET_H_
