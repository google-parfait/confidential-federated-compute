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

#include "containers/fed_sql/budget.h"

#include <algorithm>
#include <limits>
#include <string>

#include "absl/status/status_macros.h"
#include "containers/common/time_budget/budget.pb.h"

namespace confidential_federated_compute::fed_sql {

absl::Status Budget::Parse(const std::string& data) {
  BudgetState state;
  if (!state.ParseFromString(data)) {
    return absl::InternalError("Failed to parse Budget state.");
  }
  return Parse(state);
}

absl::Status Budget::Parse(const BudgetState& state) {
  InnerMap map;
  for (const auto& bucket : state.buckets()) {
    map[bucket.key()] = std::min(
        bucket.budget(),
        default_budget_.value_or(std::numeric_limits<uint32_t>::max()));
  }
  std::swap(per_key_budgets_, map);

  if (state.has_time_budget()) {
    ABSL_RETURN_IF_ERROR(time_budget_.Parse(state.time_budget()));
  }

  return absl::OkStatus();
}

std::string Budget::SerializeAsString() const {
  return Serialize().SerializeAsString();
}

BudgetState Budget::Serialize() const {
  BudgetState state;
  for (const auto& [key, budget] : per_key_budgets_) {
    auto* bucket = state.add_buckets();
    bucket->set_key(key);
    bucket->set_budget(budget);
  }
  *state.mutable_time_budget() = time_budget_.Serialize();
  return state;
}

absl::flat_hash_set<std::string> Budget::GetKeys() const {
  absl::flat_hash_set<std::string> keys;
  for (const auto& [key, _] : per_key_budgets_) {
    keys.insert(key);
  }
  return keys;
}

bool Budget::HasRemainingBudget(const std::string& key) const {
  auto it = per_key_budgets_.find(key);
  if (it == per_key_budgets_.end()) {
    // Key not found in the budget state.
    // If default_budget_ has no value, it's considered infinite.
    // Otherwise, check if the default budget is greater than 0.
    return !default_budget_.has_value() || default_budget_.value() > 0;
  }
  return it->second > 0;
}

bool Budget::HasRemainingBudget(Interval<uint64_t> time_window) {
  return time_budget_.HasRemainingBudget(time_window);
}

absl::Status Budget::UpdatePerKeyBudget(
    const absl::flat_hash_set<std::string>& keys,
    const IntervalSet<uint64_t>& ranges) {
  // A legacy per-key budget update is only valid if some ranges of data have
  // been committed.
  if (ranges.empty()) {
    return absl::FailedPreconditionError(
        "The interval set is empty when updating the budget.");
  }

  // If the default budget has no value, it means the budget is infinite.
  // In this case, there is no need to have buckets for an infinite budget.
  if (!default_budget_.has_value()) {
    return absl::OkStatus();
  }

  // First verify that all keys have remaining budget before mutating.
  for (const auto& key : keys) {
    if (!HasRemainingBudget(key)) {
      return absl::FailedPreconditionError("The budget is exhausted for key: " +
                                           key);
    }
  }

  for (const auto& key : keys) {
    // Create a bucket with the default budget if it doesn't exist.
    auto [it, _] = per_key_budgets_.try_emplace(key, default_budget_.value());
    it->second--;
  }

  return absl::OkStatus();
}

absl::Status Budget::UpdateBudget(const RangeTracker& range_tracker) {
  std::optional<Interval<uint64_t>> time_window =
      range_tracker.GetAggregationWindow();
  if (time_window.has_value()) {
    // When a time_window is provided, update only the time-based budget.
    ABSL_RETURN_IF_ERROR(time_budget_.UpdateBudget(time_window.value()));
  } else {
    // Reject legacy key-based budget updates if time-based budget is already
    // active.
    if (time_budget_.anchor_time().has_value()) {
      return absl::FailedPreconditionError(
          "Cannot update per-key budget: pipeline has already transitioned to "
          "time-based budget.");
    }
    // Otherwise, update the legacy per-key budgets.
    ABSL_RETURN_IF_ERROR(
        UpdatePerKeyBudget(range_tracker.GetKeys(), range_tracker.GetRanges()));
  }

  // Remove any expired keys from the budget.
  for (const auto& expired_key : range_tracker.GetExpiredKeys()) {
    per_key_budgets_.erase(expired_key);
  }

  return absl::OkStatus();
}

}  // namespace confidential_federated_compute::fed_sql