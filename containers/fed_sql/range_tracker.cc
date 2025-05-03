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

#include "absl/status/statusor.h"
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
  for (const auto& bucket : state.buckets()) {
    if (bucket.values_size() % 2 != 0) {
      return absl::InternalError(
          "Unexpected number of values in serialized RangeTracker state.");
    }
    std::vector<Interval<uint64_t>> intervals;
    intervals.reserve(bucket.values_size() / 2);
    auto it = bucket.values().begin();
    while (it != bucket.values().end()) {
      uint64_t start = *it++;
      uint64_t end = *it++;
      if (end <= start) {
        return absl::InternalError(
            "Unexpected order of values in serialized RangeTracker state.");
      }
      intervals.push_back(Interval(start, end));
    }
    if (!range_tracker.per_key_ranges_[bucket.key()].Assign(intervals.begin(),
                                                            intervals.end())) {
      return absl::InternalError(
          "Unexpected order of intervals in serialized RangeTracker state.");
    }
  }
  return range_tracker;
}

std::string RangeTracker::SerializeAsString() const {
  return Serialize().SerializeAsString();
}

RangeTrackerState RangeTracker::Serialize() const {
  RangeTrackerState state;
  for (const auto& [key, intervals] : per_key_ranges_) {
    auto* bucket = state.add_buckets();
    bucket->set_key(key);
    auto* values = bucket->mutable_values();
    values->Reserve(intervals.size() * 2);
    for (const auto& interval : intervals) {
      values->Add(interval.start());
      values->Add(interval.end());
    }
  }
  return state;
}

bool RangeTracker::AddRange(const std::string& key, uint64_t start,
                            uint64_t end) {
  return per_key_ranges_[key].Add(Interval(start, end));
}

bool RangeTracker::Merge(const RangeTracker& other) {
  for (const auto& [key, interval_set] : other.per_key_ranges_) {
    if (!per_key_ranges_[key].Merge(interval_set)) {
      return false;
    }
  }
  return true;
}

}  // namespace confidential_federated_compute::fed_sql