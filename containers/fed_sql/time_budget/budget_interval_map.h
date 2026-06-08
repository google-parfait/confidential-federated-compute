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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_TIME_BUDGET_BUDGET_INTERVAL_MAP_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_TIME_BUDGET_BUDGET_INTERVAL_MAP_H_

#include <cstdint>

#include "absl/log/check.h"
#include "containers/fed_sql/interval.h"
#include "containers/fed_sql/interval_map.h"

namespace confidential_federated_compute::fed_sql {

// An interval-based budget tracker backed by an IntervalMap<uint64_t,
// uint64_t>.
//
// Every point in the key space has a conceptual budget. Points not covered by
// any stored interval implicitly carry the `default_budget`. The class
// maintains the invariant that stored intervals always have a value strictly
// less than `default_budget`; intervals equal to the default are not stored
// (they are represented implicitly by the absence of an entry).
//
// Two primary operations:
//
//   HasBudget(interval): Returns true if every point in `interval` has budget >
//   0.
//
//   SubtractBudget(interval): Decrements the budget of every point in
//   `interval` by 1. Returns false without modifying the map if any point
//   already has budget == 0.
//
// Example:
//   BudgetIntervalMap map(/*default_budget=*/3);
//   map.SubtractBudget({0, 100});   // [0,100) stored as 2
//   map.SubtractBudget({50, 150});  // [0,50)→2, [50,100)→1, [100,150)→2
//   map.HasBudget({0, 150});        // true (all values > 0)
class BudgetIntervalMap {
 public:
  using InnerMap = IntervalMap<uint64_t, uint64_t>;
  using const_iterator = InnerMap::const_iterator;
  using value_type = InnerMap::value_type;

  // Equality operators.
  bool operator==(const BudgetIntervalMap& other) const {
    return default_budget_ == other.default_budget_ && map_ == other.map_;
  }
  bool operator!=(const BudgetIntervalMap& other) const {
    return !(*this == other);
  }

  // Iteration support.
  const_iterator begin() const { return map_.begin(); }
  const_iterator end() const { return map_.end(); }

  // Returns true if the map has no stored intervals.
  bool empty() const { return map_.empty(); }

  // Creates an empty BudgetIntervalMap. `default_budget` must be > 0.
  explicit BudgetIntervalMap(uint64_t default_budget)
      : default_budget_(default_budget) {
    CHECK_GT(default_budget_, 0);
  }

  // Returns true if every point in `interval` has budget > 0.
  bool HasBudget(Interval<uint64_t> interval);

  // Subtracts 1 from the budget of every point in `interval`.
  //
  // Returns false without modifying the map if any point in `interval`
  // already has budget == 0. Returns true if the interval is empty.
  bool SubtractBudget(Interval<uint64_t> interval);

  // Insert an interval in the map.
  //
  // Returns false if the value is greater than the `default_budget_`.
  // Returns false if the interval overlaps with an existing stored interval.
  bool Insert(Interval<uint64_t> interval, uint64_t value);

  // Removes expired intervals. An interval is expired if it ended more than
  // `ttl_mins` ago relative to the latest stored interval.
  void CleanupStaleIntervals(uint64_t ttl_mins);

 private:
  InnerMap map_;
  uint64_t default_budget_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_TIME_BUDGET_BUDGET_INTERVAL_MAP_H_
