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

#include "containers/fed_sql/time_budget/budget_interval_map.h"

#include <cstdint>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "containers/fed_sql/interval.h"
#include "containers/fed_sql/interval_map.h"

namespace confidential_federated_compute::fed_sql {

bool BudgetIntervalMap::HasBudget(Interval<uint64_t> interval) {
  return map_.ForEachValue(interval, [](uint64_t& v) { return v > 0; });
}

bool BudgetIntervalMap::SubtractBudget(Interval<uint64_t> interval) {
  if (interval.empty()) return true;
  if (!HasBudget(interval)) return false;

  // Decrement all stored intervals within the range.
  map_.ForEachValue(interval, [](uint64_t& v) {
    v--;
    return true;
  });

  // Fill each gap with (default_budget_ - 1)
  for (const auto& gap : map_.GetGaps(interval)) {
    CHECK(map_.Insert(gap, default_budget_ - 1)) << "Gap should never overlap";
  }
  return true;
}

bool BudgetIntervalMap::Insert(Interval<uint64_t> interval, uint64_t value) {
  // Nothing to insert in the map.
  if (interval.empty() || value == default_budget_) return true;

  if (value > default_budget_) {
    LOG(ERROR) << "Inserting value greater than default budget is not allowed.";
    return false;
  }
  return map_.Insert(interval, value);
}

void BudgetIntervalMap::CleanupStaleIntervals(uint64_t ttl_mins) {
  auto last_end = map_.last_interval_end();
  if (!last_end.has_value()) return;

  uint64_t expiration_cutoff =
      (*last_end > ttl_mins) ? *last_end - ttl_mins : 0;
  map_.EraseIf([expiration_cutoff](const auto& pair) {
    return pair.first.end() <= expiration_cutoff;
  });
}

}  // namespace confidential_federated_compute::fed_sql
