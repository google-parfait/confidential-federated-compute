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

#include "containers/fed_sql/time_budget/time_budget.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "containers/fed_sql/budget.pb.h"
#include "containers/fed_sql/interval.h"
#include "containers/fed_sql/time_budget/budget_interval_map.h"

namespace confidential_federated_compute::fed_sql {

absl::Status TimeBudget::Parse(const std::string& data) {
  BudgetState::TimeBudgetState state;
  if (!state.ParseFromString(data)) {
    return absl::InternalError("Failed to parse TimeBudgetState from string.");
  }
  return Parse(state);
}

absl::Status TimeBudget::Parse(const BudgetState::TimeBudgetState& state) {
  // If the default budget is unlimited, no need to parse the state.
  if (!default_budget_.has_value()) {
    return absl::OkStatus();
  }

  BudgetIntervalMap new_map(default_budget_.value());
  CHECK_EQ(state.anchor_time() % 60, 0)
      << "anchor_time must be minute-aligned (multiple of 60 seconds).";
  uint64_t anchor_minutes = state.anchor_time() / 60;

  for (const auto& interval : state.intervals()) {
    uint64_t start_index = interval.start_index();
    uint64_t count = interval.count();
    uint64_t remaining_budget = interval.remaining_budget();

    // Cap the parsed remaining budget to the default budget.
    remaining_budget = std::min(remaining_budget,
                                static_cast<uint64_t>(default_budget_.value()));

    // Convert relative index offsets to absolute minutes since epoch.
    uint64_t start_min = start_index + anchor_minutes;
    uint64_t end_min = start_index + count + anchor_minutes;

    Interval<uint64_t> absolute_interval(start_min, end_min);
    if (!new_map.Insert(absolute_interval, remaining_budget)) {
      return absl::InternalError(
          "Failed to parse TimeBudgetState: overlapping intervals in input "
          "state.");
    }
  }

  budget_map_ = std::move(new_map);
  return absl::OkStatus();
}

BudgetState::TimeBudgetState TimeBudget::Serialize() const {
  BudgetState::TimeBudgetState state;
  if (!default_budget_.has_value() || !budget_map_.has_value() ||
      budget_map_->empty()) {
    return state;
  }

  // If we reach here, budget_map_ must be non-empty, so
  // anchor_time() must return a valid value.
  auto anchor = anchor_time();
  CHECK(anchor.has_value());
  // Convert anchor from minutes back to seconds for serialization.
  state.set_anchor_time(*anchor * 60);

  for (const auto& [interval, remaining_budget] : *budget_map_) {
    // The map stores minutes, so index offsets are direct differences.
    uint64_t start_index = interval.start() - *anchor;
    uint64_t count = interval.end() - interval.start();

    auto* proto_interval = state.add_intervals();
    proto_interval->set_start_index(start_index);
    proto_interval->set_count(count);
    proto_interval->set_remaining_budget(remaining_budget);
  }
  return state;
}

std::string TimeBudget::SerializeAsString() const {
  return Serialize().SerializeAsString();
}

bool TimeBudget::HasRemainingBudget(Interval<uint64_t> time_window) {
  // Unlimited budget case, return true.
  if (!default_budget_.has_value()) {
    return true;
  }

  // Empty input window - no time needs to be consumed, so return true.
  if (time_window.empty()) {
    return true;
  }

  // Convert input seconds to minute boundaries.
  uint64_t start_min = time_window.start() / 60;
  uint64_t end_min = (time_window.end() + 59) / 60;

  auto anchor = anchor_time();
  if (anchor.has_value() && start_min < *anchor) {
    return false;
  }

  return budget_map_->HasBudget(Interval<uint64_t>(start_min, end_min));
}

absl::Status TimeBudget::UpdateBudget(Interval<uint64_t> time_window) {
  // If budget is unlimited, there's nothing to do.
  if (!default_budget_.has_value()) {
    return absl::OkStatus();
  }

  // If the input time window is empty, there's nothing to do.
  if (time_window.empty()) {
    return absl::OkStatus();
  }

  // Convert input seconds to minute boundaries.
  uint64_t start_min = time_window.start() / 60;
  uint64_t end_min = (time_window.end() + 59) / 60;

  auto anchor = anchor_time();
  if (anchor.has_value() && start_min < *anchor) {
    return absl::FailedPreconditionError(
        "Cannot update budget for a time window starting before the anchor "
        "time.");
  }

  Interval<uint64_t> aligned_window(start_min, end_min);
  if (!budget_map_->SubtractBudget(aligned_window)) {
    return absl::FailedPreconditionError(
        "The budget is exhausted for the specified time window.");
  }

  return absl::OkStatus();
}

}  // namespace confidential_federated_compute::fed_sql
