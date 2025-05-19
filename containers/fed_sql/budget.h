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

  // Checks whether any budget remains for the specified key.
  bool HasRemainingBudget(const std::string& key);

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