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

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "containers/fed_sql/range_tracker.pb.h"
#include "fcp/base/monitoring.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

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
  range_tracker.expired_keys_.insert(state.expired_keys().begin(),
                                     state.expired_keys().end());
  range_tracker.partition_index_ = state.partition_index();
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

  for (const auto& key : expired_keys_) {
    state.add_expired_keys(key);
  }

  state.set_partition_index(partition_index_);
  return state;
}

bool RangeTracker::AddRange(const std::string& key, uint64_t start,
                            uint64_t end) {
  // Key must not be expired.
  CHECK(!expired_keys_.contains(key)) << "Found an expired key " << key;
  return per_key_ranges_[key].Add(Interval(start, end));
}

bool RangeTracker::Merge(const RangeTracker& other) {
  if (partition_index_ != other.partition_index_) {
    LOG(ERROR) << "Attempting to merge RangeTrackers with different partition "
                  "indices: "
               << partition_index_ << " and " << other.partition_index_;
    return false;
  }

  if (expired_keys_ != other.expired_keys_) {
    LOG(ERROR) << "Expired keys for RangeTracker are not identical";
    return false;
  }

  for (const auto& [key, interval_set] : other.per_key_ranges_) {
    if (!per_key_ranges_[key].Merge(interval_set)) {
      return false;
    }
  }
  return true;
}

std::string BundleRangeTracker(std::string blob,
                               const RangeTracker& range_tracker) {
  std::string buffer;
  google::protobuf::io::StringOutputStream stream(&buffer);
  google::protobuf::io::CodedOutputStream coded_stream(&stream);
  coded_stream.WriteString(kRangeTrackerBundleSignature);
  std::string range_tracker_state = range_tracker.SerializeAsString();
  coded_stream.WriteVarint64(range_tracker_state.size());
  coded_stream.WriteString(range_tracker_state);
  coded_stream.WriteVarint64(blob.size());
  coded_stream.Trim();
  buffer.append(blob);
  return buffer;
}

absl::StatusOr<RangeTracker> UnbundleRangeTracker(std::string& blob) {
  google::protobuf::io::ArrayInputStream stream(blob.data(), blob.size());
  google::protobuf::io::CodedInputStream coded_stream(&stream);
  std::string signature;
  if (!coded_stream.ReadString(&signature,
                               sizeof(kRangeTrackerBundleSignature) - 1) ||
      signature != kRangeTrackerBundleSignature) {
    return absl::InvalidArgumentError(
        "Invalid input blob: RangeTracker bundle signature mismatch");
  }

  size_t range_tracker_state_size;
  if (!coded_stream.ReadVarint64(&range_tracker_state_size)) {
    return absl::InvalidArgumentError(
        "Invalid input blob: RangeTracker state size is missing");
  }

  std::string range_tracker_state;
  if (!coded_stream.ReadString(&range_tracker_state,
                               range_tracker_state_size)) {
    return absl::InvalidArgumentError(
        "Invalid input blob: insufficient RangeTracker state");
  }

  FCP_ASSIGN_OR_RETURN(RangeTracker range_tracker,
                       RangeTracker::Parse(range_tracker_state));

  size_t payload_size;
  if (!coded_stream.ReadVarint64(&payload_size)) {
    return absl::InvalidArgumentError(
        "Invalid input blob: payload size is missing");
  }

  int pos = coded_stream.CurrentPosition();

  if (pos + payload_size != blob.size()) {
    return absl::InvalidArgumentError("Invalid input blob: incomplete payload");
  }

  // Make a new blob for the remaining "payload" part of the original blob.
  blob = blob.substr(pos);
  return range_tracker;
}

}  // namespace confidential_federated_compute::fed_sql