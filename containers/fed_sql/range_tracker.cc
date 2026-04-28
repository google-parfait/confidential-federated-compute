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
#include "google/protobuf/text_format.h"

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

bool RangeTracker::Merge(const RangeTracker& other) {
  // Merge keys.
  keys_.insert(other.keys_.begin(), other.keys_.end());

  // Merge ranges (if there is no overlap)
  if (!ranges_.Merge(other.ranges_)) {
    return false;
  }

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

  // Merge expired keys. If a key expires while a pipeline is still running,
  // it's possible that different partitions hold different
  // `expired_keys_` so we merge the sets here.
  expired_keys_.insert(other.expired_keys_.begin(), other.expired_keys_.end());
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