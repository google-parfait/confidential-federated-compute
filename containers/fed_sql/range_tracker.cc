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

#include "containers/fed_sql/range_tracker.h"

#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "containers/fed_sql/any_bundle.h"
#include "containers/fed_sql/range_tracker.pb.h"

namespace confidential_federated_compute::fed_sql {

absl::StatusOr<RangeTracker> RangeTracker::Parse(const std::string& data) {
  RangeTrackerState state;
  if (!state.ParseFromString(data)) {
    return absl::InternalError("Failed to parse RangeTracker state.");
  }
  return Parse(state);
}

absl::StatusOr<RangeTracker> RangeTracker::Parse(
    const RangeTrackerState& state) {
  RangeTracker range_tracker;
  range_tracker.keys_ = absl::flat_hash_set<std::string>(state.keys().begin(),
                                                         state.keys().end());

  if (state.values_size() % 2 != 0) {
    return absl::InternalError(
        "Unexpected number of values in serialized RangeTracker state.");
  }
  std::vector<Interval<uint64_t>> intervals;
  intervals.reserve(state.values_size() / 2);
  auto it = state.values().begin();
  while (it != state.values().end()) {
    uint64_t start = *it++;
    uint64_t end = *it++;
    if (end <= start) {
      return absl::InternalError(
          "Unexpected order of values in serialized RangeTracker state.");
    }
    intervals.push_back(Interval(start, end));
  }
  if (!range_tracker.ranges_.Assign(intervals.begin(), intervals.end())) {
    return absl::InternalError(
        "Unexpected order of intervals in serialized RangeTracker state.");
  }

  range_tracker.expired_keys_ = absl::flat_hash_set<std::string>(
      state.expired_keys().begin(), state.expired_keys().end());
  range_tracker.partition_index_ = state.partition_index();

  if (state.has_start_time() != state.has_end_time()) {
    return absl::InternalError(
        "RangeTracker state must have either both start_time and end_time or "
        "neither.");
  }
  if (state.has_start_time()) {
    range_tracker.agg_window_ = Interval<uint64_t>(state.start_time().seconds(),
                                                   state.end_time().seconds());
  }

  return range_tracker;
}

std::string RangeTracker::SerializeAsString() const {
  return Serialize().SerializeAsString();
}

RangeTrackerState RangeTracker::Serialize() const {
  RangeTrackerState state;
  for (const auto& key : keys_) {
    state.add_keys(key);
  }
  for (const auto& interval : ranges_) {
    state.add_values(interval.start());
    state.add_values(interval.end());
  }
  for (const auto& key : expired_keys_) {
    state.add_expired_keys(key);
  }
  if (partition_index_.has_value()) {
    state.set_partition_index(partition_index_.value());
  }
  if (agg_window_.has_value()) {
    state.mutable_start_time()->set_seconds(agg_window_->start());
    state.mutable_end_time()->set_seconds(agg_window_->end());
  }
  return state;
}

void RangeTracker::AddKey(const std::string& key) {
  // Key must not be expired.
  CHECK(!expired_keys_.contains(key)) << "Found an expired key " << key;
  keys_.insert(key);
}

bool RangeTracker::AddRange(uint64_t start, uint64_t end) {
  return ranges_.Add(Interval(start, end));
}

void RangeTracker::MergeAggWindow(Interval<uint64_t> agg_window) {
  if (!agg_window_.has_value()) {
    agg_window_ = agg_window;
    return;
  }

  // Create a bounding aggregation window (smallest start, largest end).
  agg_window_ =
      Interval<uint64_t>(std::min(agg_window_->start(), agg_window.start()),
                         std::max(agg_window_->end(), agg_window.end()));
}

bool RangeTracker::Merge(const RangeTracker& other) {
  // Partition keys must match (if both are set).
  if (!partition_index_.has_value()) {
    partition_index_ = other.partition_index_;
  } else if (other.partition_index_.has_value() &&
             partition_index_ != other.partition_index_) {
    LOG(ERROR) << "Attempting to merge RangeTrackers with different partition "
                  "indices: "
               << *partition_index_ << " and " << *other.partition_index_;
    return false;
  }

  return Merge(other.keys_, other.ranges_, other.expired_keys_,
               other.agg_window_);
}

bool RangeTracker::Merge(const absl::flat_hash_set<std::string>& keys,
                         const IntervalSet<uint64_t>& ranges,
                         const absl::flat_hash_set<std::string>& expired_keys,
                         std::optional<Interval<uint64_t>> agg_window) {
  if (agg_window.has_value()) {
    MergeAggWindow(agg_window.value());
  }

  // Merge keys.
  keys_.insert(keys.begin(), keys.end());

  // Merge ranges (if there is no overlap)
  if (!ranges_.Merge(ranges)) {
    return false;
  }

  // Merge expired keys. If a key expires while a pipeline is still running,
  // it's possible that different partitions hold different
  // `expired_keys_` so we merge the sets here.
  expired_keys_.insert(expired_keys.begin(), expired_keys.end());
  return true;
}

std::string BundleRangeTracker(std::string blob,
                               const RangeTracker& range_tracker) {
  return std::string(
      BundleAny(range_tracker.Serialize(), absl::Cord(std::move(blob))));
}

absl::StatusOr<RangeTracker> UnbundleRangeTracker(std::string& blob) {
  RangeTrackerState state;
  absl::Cord data(std::move(blob));
  if (!UnbundleAny(state, data)) {
    return absl::InvalidArgumentError("Failed to unbundle RangeTracker");
  }

  blob = std::string(std::move(data));
  return RangeTracker::Parse(state);
}

}  // namespace confidential_federated_compute::fed_sql