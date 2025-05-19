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
    map[bucket.key()] = std::min(
        bucket.budget(),
        default_budget_.value_or(std::numeric_limits<uint32_t>::max()));
  }
  std::swap(per_key_budgets_, map);
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
  return state;
}

bool Budget::HasRemainingBudget(const std::string& key) {
  auto it = per_key_budgets_.find(key);
  uint32_t budget =
      it != per_key_budgets_.end()
          ? it->second
          : default_budget_.value_or(std::numeric_limits<uint32_t>::max());
  return budget > 0;
}

absl::Status Budget::UpdateBudget(const RangeTracker& range_tracker) {
  for (const auto& [key, unused] : range_tracker) {
    InnerMap::iterator it = per_key_budgets_.find(key);
    if (it == per_key_budgets_.end() && default_budget_.has_value()) {
      // Create a bucket if it doesn't exist and if the default budget
      // isn't infinite. If the default budget is infinite. There is no
      // need to have a bucket for a infinite budget.
      it = per_key_budgets_.emplace(key, default_budget_.value()).first;
    }

    if (it != per_key_budgets_.end()) {
      if (it->second == 0) {
        return absl::FailedPreconditionError("The budget is exhausted");
      }

      // Consume the budget in this bucket. It is doen't matter what ranges
      // have been tracked in the RangeTracker. If the bucket exists in the
      // RangeTracker we assume that all blobs mapped to this bucket have
      // been accessed even if that isn't actually the case. This approach
      // is conservative, but it allows a very small budget state.
      it->second--;
    } else {
      // This should be the case only if the budget is infinite.
      CHECK(!default_budget_.has_value());
    }
  }
  return absl::OkStatus();
}

}  // namespace confidential_federated_compute::fed_sql
