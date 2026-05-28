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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_BUDGET_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_BUDGET_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "containers/fed_sql/budget.pb.h"
#include "containers/fed_sql/range_tracker.h"
#include "containers/fed_sql/time_budget/time_budget.h"

namespace confidential_federated_compute::fed_sql {

// BudgetInfo stores information about the budget for a single bucket.
// The total remaining budget for a key is best understood as a combination
// of the integer `budget` value and the optional `consumed_range`.
//
// - `budget`: An integer representing the number of *full* remaining uses.
// - `consumed_range`: This field defines a single contiguous range
//   [consumed_range.start, consumed_range.end) that has been *partially*
//   consumed. The remaining portion of the range is considered to have an
//   *additional* budget of 1. This is because the next use of the range will be
//   allowed to consume the last remaining portion of the range. The start is
//   inclusive and the end is exclusive.
//
// The conceptual remaining budget is the integer `budget` value plus the
// fraction of the range *not* covered by the `consumed_range`.
//
// Examples:
// 1. {budget: 5, consumed_range.start: not set, consumed_range.end: not set}:
//    Exactly 5 full uses remain.
// 2. {budget: 2, consumed_range.start: 10, consumed_range.end: 100}:
//    There are 2 full uses remaining, PLUS the portions of the range
//    outside of [10, 100) from an additional budget unit. This can be
//    thought of as "2 and a fraction" budget remaining.
// 3. {budget: 0, consumed_range.start: 10, consumed_range.end: 100}:
//    The budget is not fully exhausted. Only the range [10, 100) has been
//    consumed from the last unit of budget. Access to keys outside this
//    range is still permitted.
// 4. {budget: 0, consumed_range.start: not set, consumed_range.end: not set}:
//    The budget is fully exhausted.
//
// When a new range of data is accessed:
// - If the new range overlaps with the existing `consumed_range`, or if
//   no `consumed_range` was previously set, the integer `budget` is
//   decremented by 1. The new `consumed_range` becomes the range of the
//   data just accessed. This is a good approximation to simplify
//   implementation. We are consuming one unit of budget for the whole range
//   of data just accessed if there is any overlap because it simplifies the
//   logic.
// - If the new range does NOT overlap, the `budget` remains unchanged,
//   and the `consumed_range` is expanded to become the minimum bounding
//   range covering both the old and the new ranges. The consumed_range may be
//   a wider range than strictly necessary, but it simplifies the
//   implementation and budget tracking.
// - If a `consumed_range` expands to cover the entire range, the
//   `consumed_range` fields are reset, as a full budget unit has been
//   consumed.
struct BudgetInfo {
  uint32_t budget;
  std::optional<Interval<uint64_t>> consumed_range;

  BudgetInfo() : budget(0) {}
  explicit BudgetInfo(uint32_t b) : budget(b) {}

  bool operator==(const BudgetInfo& other) const {
    return budget == other.budget && consumed_range == other.consumed_range;
  }
};

// Budget class serves as a migration layer between the legacy per-key budget
// tracking and the new time-based budget tracking (TimeBudget). It maintains
// both budget schemes in parallel:
//
// - Legacy per-key budgets: Tracked via per_key_budgets_, using KMS key
//   IDs as bucket identifiers.
// - Time-based budget: Tracked via time_budget_, using time windows.
//
// During the migration, when a time_window is provided to UpdateBudget, only
// the TimeBudget is updated (per_key_budgets_ are not modified). When no
// time_window is provided, only the legacy per_key_budgets_ are updated.
// Expired keys are always cleaned up from per_key_budgets_ regardless.
//
// Once all callers have fully switched to time-based budgets, this class will
// become redundant and can be deleted in favor of using TimeBudget directly.
class Budget {
 public:
  using InnerMap = absl::flat_hash_map<std::string, BudgetInfo>;
  using const_iterator = typename InnerMap::const_iterator;
  using value_type = typename InnerMap::value_type;

  // Creates an instance of Budget with the specified default budget value.
  // If specified, the default budget is used for any new buckets that haven't
  // been seen before. Also, the default budget is used as the upper limit when
  // parsing a previously serialized budget in case of the policy change.
  // If the default budget isn't specified (std::nullopt), it is considered to
  // be infinite.
  explicit Budget(std::optional<uint32_t> default_budget)
      : default_budget_(default_budget), time_budget_(default_budget) {}

  // This class is move-only.
  Budget(const Budget&) = delete;
  Budget& operator=(const Budget&) = delete;

  Budget(Budget&&) = default;
  Budget& operator=(Budget&&) = default;

  // Deserializes Budget from the BudgetState message.
  absl::Status Parse(const BudgetState& state);

  // Deserializes Budget from a string buffer.
  absl::Status Parse(const std::string& data);

  // Serializes the current budget to a BudgetState message.
  BudgetState Serialize() const;

  // Serializes the current budget to a string.
  std::string SerializeAsString() const;

  // Checks whether any budget remains for the specified bucket key and
  // range_key.
  bool HasRemainingBudget(const std::string& key, uint64_t range_key);

  // Checks whether any time-based budget remains for the specified time window.
  bool HasRemainingBudget(Interval<uint64_t> time_window);

  // Checks whether the default budget is unlimited.
  bool HasUnlimitedBudget() const { return !default_budget_.has_value(); }

  // Update the budget by applying the data collected in the RangeTracker.
  // If the RangeTracker has an aggregation window set, only the time-based
  // budget is updated (per_key_budgets_ are not modified). Expired keys are
  // always cleaned up.
  absl::Status UpdateBudget(const RangeTracker& range_tracker);

  // Gets all the keys in the budget.
  absl::flat_hash_set<std::string> GetKeys() const;

  // Iteration support.
  const_iterator begin() const { return per_key_budgets_.begin(); }
  const_iterator end() const { return per_key_budgets_.end(); }

 private:
  // Updates the legacy per-key budgets based on the provided keys and ranges.
  absl::Status UpdatePerKeyBudget(const absl::flat_hash_set<std::string>& keys,
                                  const IntervalSet<uint64_t>& ranges);

  std::optional<uint32_t> default_budget_;
  // Stores budgets for individual buckets organized by key_id of encryption
  // key used to encrypt a blob (the same key_id that is found in BlobHeader).
  InnerMap per_key_budgets_;
  // Time-based budget tracking.
  TimeBudget time_budget_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_BUDGET_H_