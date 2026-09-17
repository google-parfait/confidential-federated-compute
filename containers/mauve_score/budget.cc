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
#include "budget.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "containers/common/intervals/interval.h"
#include "containers/common/time_budget/budget.pb.h"
#include "containers/common/time_budget/time_budget.h"

namespace confidential_federated_compute::mauve_score {

absl::StatusOr<Budget> Budget::Create(std::optional<std::string> initial_state,
                                      uint32_t default_budget) {
  if (default_budget == 0) {
    return absl::ResourceExhaustedError(
        "Budget exhausted: default budget must be greater than zero.");
  }

  absl::flat_hash_map<std::string, uint32_t> per_key_budgets;
  TimeBudget time_budget(default_budget);

  if (initial_state.has_value() && !initial_state->empty()) {
    BudgetState state;
    if (!state.ParseFromString(*initial_state)) {
      return absl::InvalidArgumentError(
          "Failed to parse pipeline state as BudgetState.");
    }
    for (const auto& bucket : state.buckets()) {
      per_key_budgets[bucket.key()] = std::min(bucket.budget(), default_budget);
    }
    if (state.has_time_budget()) {
      ABSL_RETURN_IF_ERROR(time_budget.Parse(state.time_budget()));
    }
  }

  return Budget(std::move(initial_state), default_budget,
                std::move(per_key_budgets), std::move(time_budget));
}

bool Budget::HasRemainingBudget(const std::string& key) const {
  auto it = per_key_budgets_.find(key);
  if (it == per_key_budgets_.end()) {
    return default_budget_ > 0;
  }
  return it->second > 0;
}

bool Budget::HasRemainingBudget(Interval<uint64_t> time_window) {
  return time_budget_.HasRemainingBudget(time_window);
}

absl::Status Budget::UpdatePerKeyBudget(
    const absl::flat_hash_set<std::string>& keys) {
  if (time_budget_.anchor_time().has_value()) {
    return absl::FailedPreconditionError(
        "Cannot update per-key budget: pipeline has already transitioned to "
        "time-based budget.");
  }

  // First verify all keys have remaining budget before mutating.
  for (const auto& key : keys) {
    if (!HasRemainingBudget(key)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "The budget is exhausted for key: ", absl::BytesToHexString(key)));
    }
  }

  for (const auto& key : keys) {
    auto it = per_key_budgets_.find(key);
    if (it == per_key_budgets_.end()) {
      it = per_key_budgets_.emplace(key, default_budget_).first;
    }
    it->second--;
  }

  return absl::OkStatus();
}

absl::Status Budget::UpdateTimeBudget(Interval<uint64_t> time_window) {
  if (!HasRemainingBudget(time_window)) {
    return absl::FailedPreconditionError(
        absl::StrCat("No time-window budget remaining for interval: [",
                     time_window.start(), ", ", time_window.end(), ")"));
  }

  return time_budget_.UpdateBudget(time_window);
}

BudgetState Budget::Serialize() const {
  BudgetState state;

  for (const auto& [key, value] : per_key_budgets_) {
    auto* bucket = state.add_buckets();
    bucket->set_key(key);
    bucket->set_budget(value);
  }
  *state.mutable_time_budget() = time_budget_.Serialize();
  return state;
}

std::string Budget::SerializeAsString() const {
  return Serialize().SerializeAsString();
}

}  // namespace confidential_federated_compute::mauve_score
