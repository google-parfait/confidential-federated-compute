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
#include "absl/status/status.h"
#include "containers/fed_sql/budget.pb.h"
#include "containers/fed_sql/range_tracker.h"

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

// Budget class is used to track and update serialized budget for FedSql
// workloads. The budget is represented as a collection of buckets associated
// with encryption keys, with each bucket having its own remaining budget.
// Each customer uploaded blob can be associated with at most one bucket,
// based on which encryption key that blob was encrypted with at upload.
// When processing blobs in a container an assumption is made that all blobs
// associated with a bucket (i.e. the corresponding encryption key) have been
//  processed, therefore the budget for that bucket is reduced by 1.

// TODO: Add ability to remove expired buckets from the budget
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
      : default_budget_(default_budget) {}

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

  // Update the budget by applying the data collected in the RangeTracker.
  absl::Status UpdateBudget(const RangeTracker& range_tracker);

  // Iteration support.
  const_iterator begin() const { return per_key_budgets_.begin(); }
  const_iterator end() const { return per_key_budgets_.end(); }

 private:
  std::optional<uint32_t> default_budget_;
  // Stores budgets for individual buckets organized by key_id of encryption
  // key used to encrypt a blob (the same key_id that is found in BlobHeader).
  InnerMap per_key_budgets_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_BUDGET_H_