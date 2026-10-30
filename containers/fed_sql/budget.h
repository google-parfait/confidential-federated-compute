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
#include "containers/common/time_budget/budget.pb.h"
#include "containers/common/time_budget/time_budget.h"
#include "containers/fed_sql/range_tracker.h"

namespace confidential_federated_compute::fed_sql {

// Budget class serves as a migration layer between the legacy per-key budget
// tracking and the new time-based budget tracking (TimeBudget). It maintains
// both budget schemes in parallel:
//
// - Legacy per-key budgets: Tracked via per_key_budgets_, using KMS key
//   IDs as bucket identifiers.
// - Time-based budget: Tracked via time_budget_, using time windows.
//
// During the migration, when a time_window is provided to UpdateBudget, only
// the TimeBudget is updated (per_key_budgets_ are not modified). Once
// time-based budget has been consumed, legacy per-key budget updates are
// disallowed to prevent double-consumption of blobs.
// If time-based budget is not consumed, the legacy per_key_budgets_ are
// updated as normal. Expired keys are always cleaned up from per_key_budgets_
// regardless.
//
// Once all callers have fully switched to time-based budgets, this class will
// become redundant and can be deleted in favor of using TimeBudget directly.
class Budget {
 public:
  // Maps a bucket key to the number of remaining uses for that bucket.
  using InnerMap = absl::flat_hash_map<std::string, uint32_t>;
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

  // Checks whether any budget remains for the specified bucket key.
  bool HasRemainingBudget(const std::string& key) const;

  // Checks whether any time-based budget remains for the specified time window.
  bool HasRemainingBudget(Interval<uint64_t> time_window);

  // Checks whether the default budget is unlimited.
  bool HasUnlimitedBudget() const { return !default_budget_.has_value(); }

  // Update the budget by applying the data collected in the RangeTracker.
  // If the RangeTracker has an aggregation window set, only the time-based
  // budget is updated (per_key_budgets_ are not modified). If no aggregation
  // window is set, updates the legacy per-key budgets unless time-based budget
  // has already been consumed, in which case an error is returned. Expired keys
  // are always cleaned up.
  absl::Status UpdateBudget(const RangeTracker& range_tracker);

  // Gets all the keys in the budget.
  absl::flat_hash_set<std::string> GetKeys() const;

  // Iteration support.
  const_iterator begin() const { return per_key_budgets_.begin(); }
  const_iterator end() const { return per_key_budgets_.end(); }

 private:
  // Updates the legacy per-key budgets by consuming one unit of budget for
  // each of the provided keys. The ranges must be non-empty but otherwise
  // don't affect how much budget is consumed. Fails without modifying the
  // budget if any of the keys has no remaining budget.
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