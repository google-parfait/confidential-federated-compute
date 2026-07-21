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
    uint32_t budget_val = std::min(
        bucket.budget(),
        default_budget_.value_or(std::numeric_limits<uint32_t>::max()));

    if (bucket.has_consumed_range_start() != bucket.has_consumed_range_end()) {
      return absl::InternalError(
          "Failed to parse Budget state: consumed_range_start and "
          "consumed_range_end must both be specified or neither.");
    }

    BudgetInfo info(budget_val);
    if (bucket.has_consumed_range_start() && bucket.has_consumed_range_end()) {
      info.consumed_range = Interval<uint64_t>(bucket.consumed_range_start(),
                                               bucket.consumed_range_end());
    }
    map[bucket.key()] = info;
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
  for (const auto& [key, info] : per_key_budgets_) {
    auto* bucket = state.add_buckets();
    bucket->set_key(key);
    bucket->set_budget(info.budget);
    if (info.consumed_range.has_value()) {
      bucket->set_consumed_range_start(info.consumed_range->start());
      bucket->set_consumed_range_end(info.consumed_range->end());
    }
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

bool Budget::HasRemainingBudget(const std::string& key, uint64_t range_key) {
  auto it = per_key_budgets_.find(key);

  if (it == per_key_budgets_.end()) {
    // Key not found in the budget state.
    // If default_budget_ has no value, it's considered infinite.
    // Otherwise, check if the default budget is greater than 0.
    return !default_budget_.has_value() || default_budget_.value() > 0;
  }

  const BudgetInfo& budget_info = it->second;

  if (budget_info.budget > 0) {
    // If the remaining budget is greater than 1, the blob is always allowed,
    // as at least one full usage remains regardless of any partial consumption.
    return true;
  }

  if (budget_info.budget == 0) {
    // If budget is exactly 0, we must check the partially consumed range.
    if (budget_info.consumed_range.has_value()) {
      // A partially consumed range exists. The blob is allowed only if it falls
      // *outside* this range.
      bool is_outside_consumed_range =
          range_key < budget_info.consumed_range->start() ||
          range_key >= budget_info.consumed_range->end();
      return is_outside_consumed_range;
    }
  }

  // Budget is 0. No access is allowed.
  return false;
}

bool Budget::HasRemainingBudget(Interval<uint64_t> time_window) {
  return time_budget_.HasRemainingBudget(time_window);
}

absl::Status Budget::UpdatePerKeyBudget(
    const absl::flat_hash_set<std::string>& keys,
    const IntervalSet<uint64_t>& ranges) {
  if (ranges.empty()) {
    return absl::FailedPreconditionError(
        "The interval set is empty when updating the budget.");
  }

  // The same ranges apply to all keys.
  uint64_t interval_min = ranges.BoundingInterval().start();
  uint64_t interval_max = ranges.BoundingInterval().end();

  for (const auto& key : keys) {
    // If the default budget has no value, it means the budget is infinite.
    // In this case, there is no need to have a bucket for a infinite budget.
    if (!default_budget_.has_value()) {
      continue;
    }

    auto it = per_key_budgets_.find(key);
    if (it == per_key_budgets_.end()) {
      // Create a bucket if it doesn't exist.
      it = per_key_budgets_.emplace(key, BudgetInfo(default_budget_.value()))
               .first;
    }

    BudgetInfo& budget_info = it->second;

    if (!budget_info.consumed_range.has_value()) {
      // If there is no partially consumed interval, we can think about it as
      // the last budget unit has no partial range available. So we can just
      // subtract the budget and set the new consumed range to the current one.
      if (budget_info.budget == 0) {
        return absl::FailedPreconditionError(
            "The budget is exhausted for key: " + key);
      }
      budget_info.budget--;
      budget_info.consumed_range =
          Interval<uint64_t>(interval_min, interval_max);
    } else {
      const uint64_t old_interval_start = budget_info.consumed_range->start();
      const uint64_t old_interval_end = budget_info.consumed_range->end();
      // Check for any overlap between [interval_min, interval_max) and
      // [old_interval_start, old_interval_end).
      const bool ranges_overlap =
          interval_max > old_interval_start && interval_min < old_interval_end;

      if (ranges_overlap) {
        if (budget_info.budget == 0) {
          return absl::FailedPreconditionError(
              "The budget is exhausted for key: " + key);
        }
        budget_info.budget--;
        // If there is an overlap - we subtract the budget by 1 and set the new
        // interval to the intersection.
        budget_info.consumed_range =
            Interval<uint64_t>(std::max(old_interval_start, interval_min),
                               std::min(old_interval_end, interval_max));
      } else {
        // If there is no overlap - we do the union and don't subtract the
        // budget.
        budget_info.consumed_range =
            Interval<uint64_t>(std::min(old_interval_start, interval_min),
                               std::max(old_interval_end, interval_max));
      }
    }

    if (budget_info.consumed_range->start() == 0 and
        budget_info.consumed_range->end() ==
            std::numeric_limits<uint64_t>::max()) {
      // The range covered the entire set, clear any partial range tracking.
      budget_info.consumed_range.reset();
    }
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