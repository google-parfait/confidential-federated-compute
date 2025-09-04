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

#include "containers/fed_sql/budget.pb.h"

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
  return state;
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

absl::Status Budget::UpdateBudget(const RangeTracker& range_tracker) {
  for (const auto& [key, interval_set] : range_tracker) {
    if (interval_set.empty()) {
      return absl::FailedPreconditionError(
          "The interval set is empty for key: " + key);
    }
    // If the default budget has no value, it means the budget is infinite.
    // In this case, there is no need to have a bucket for a infinite budget.
    if (!default_budget_.has_value()) {
      continue;
    }

    uint64_t interval_min = interval_set.BoundingInterval().start();
    uint64_t interval_max = interval_set.BoundingInterval().end();

    auto it = per_key_budgets_.find(key);
    if (it == per_key_budgets_.end()) {
      // Create a bucket if it doesn't exist.
      it = per_key_budgets_.emplace(key, BudgetInfo(default_budget_.value()))
               .first;
    }

    BudgetInfo& budget_info = it->second;

    bool has_old_range = budget_info.consumed_range.has_value();

    bool ranges_overlap = false;
    if (budget_info.budget == 0 and !has_old_range) {
      return absl::FailedPreconditionError("The budget is exhausted for key: " +
                                           key);
    }
    if (has_old_range) {
      // Check for any overlap between [interval_min, interval_max) and
      // [old_range_start, old_range_end).
      ranges_overlap = interval_max > budget_info.consumed_range->start() &&
                       interval_min < budget_info.consumed_range->end();
    }
    if (budget_info.budget == 0 and ranges_overlap) {
      return absl::FailedPreconditionError("The budget is exhausted for key: " +
                                           key);
    }

    // The budget will be decremented by 1 if there were no
    // `consumed_range_start` and `consumed_range_end` previously, or if the
    // bounded partial range consumed overlaps with the `consumed_range_start`
    // and `consumed_range_end` of the previous budget.
    bool should_decrement = !has_old_range || ranges_overlap;

    if (should_decrement) {
      CHECK_GT(budget_info.budget, 0) << "Budget must be positive.";
      budget_info.budget--;
      // Since `should_decrement` is true, we are now tracking the
      // fractional consumption on the *next* available budget unit. This
      // was triggered because either:
      // 1. There was no previous `consumed_range` (the last budget unit
      //    had no partial range available).
      // 2. The current access range [interval_min, interval_max) overlapped
      //    with the previous `consumed_range`, meaning to appropriately track
      //    the budget of that range, we needed to clear the previous
      //    `consumed_range` and start tracking the fractional consumption on
      //    the next budget unit.
      //
      // Therefore, the new `consumed_range` for this fresh budget unit
      // starts with the current access range [interval_min, interval_max).
      budget_info.consumed_range =
          Interval<uint64_t>(interval_min, interval_max);

    } else {  // Has old range and NO overlap
      // Extend the existing consumed range to cover the new range.
      budget_info.consumed_range = Interval<uint64_t>(
          std::min(interval_min, budget_info.consumed_range->start()),
          std::max(interval_max, budget_info.consumed_range->end()));
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

}  // namespace confidential_federated_compute::fed_sql
