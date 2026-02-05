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

#include "containers/fed_sql/partition_private_state.h"

#include "absl/log/log.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"

namespace confidential_federated_compute::fed_sql {

using ::fcp::confidentialcompute::FedSqlContainerPartitionKeys;

absl::StatusOr<PartitionPrivateState> PartitionPrivateState::Parse(
    const std::string& data) {
  PartitionPrivateStateProto state;
  if (!state.ParseFromString(data)) {
    return absl::InvalidArgumentError("Failed to parse PartitionPrivateState.");
  }
  return Parse(state);
}

absl::StatusOr<PartitionPrivateState> PartitionPrivateState::Parse(
    const PartitionPrivateStateProto& proto) {
  PartitionPrivateState state;
  for (const auto& entry : proto.symmetric_keys()) {
    state.symmetric_keys_.insert({entry.id(), entry.symmetric_key()});
  }

  state.expired_keys_.insert(proto.expired_keys().begin(),
                             proto.expired_keys().end());

  for (const auto& bucket : proto.buckets()) {
    // Each interval is represented by a pair of values (start, end) so it must
    // be even.
    if (bucket.values_size() % 2 != 0) {
      return absl::InvalidArgumentError(
          "Unexpected number of values in serialized PartitionPrivateState.");
    }
    std::vector<Interval<uint64_t>> intervals;
    intervals.reserve(bucket.values_size() / 2);
    for (int i = 0; i < bucket.values_size(); i += 2) {
      intervals.emplace_back(bucket.values(i), bucket.values(i + 1));
    }
    state.per_key_ranges_[bucket.key()].Assign(intervals.begin(),
                                               intervals.end());
  }
  return state;
}

std::string PartitionPrivateState::SerializeAsString() const {
  return Serialize().SerializeAsString();
}

PartitionPrivateStateProto PartitionPrivateState::Serialize() const {
  PartitionPrivateStateProto proto;
  for (const auto& [id, symmetric_key] : symmetric_keys_) {
    auto* entry = proto.add_symmetric_keys();
    entry->set_id(id);
    entry->set_symmetric_key(symmetric_key);
  }
  for (const auto& key : expired_keys_) {
    proto.add_expired_keys(key);
  }
  for (const auto& [key, intervals] : per_key_ranges_) {
    auto* bucket = proto.add_buckets();
    bucket->set_key(key);
    auto* values = bucket->mutable_values();
    values->Reserve(intervals.size() * 2);
    for (const auto& interval : intervals) {
      values->Add(interval.start());
      values->Add(interval.end());
    }
  }
  return proto;
}

bool PartitionPrivateState::AddPartition(const RangeTracker& range_tracker,
                                         absl::string_view symmetric_key) {
  // Validate that no partition ids overlap.
  std::optional<uint64_t> partition_index = range_tracker.GetPartitionIndex();
  if (!partition_index.has_value()) {
    LOG(ERROR) << "RangeTracker must have a partition index to add it "
                  "to PartitionPrivateState.";
    return false;
  }
  if (symmetric_keys_.contains(*partition_index)) {
    LOG(ERROR) << "PartitionPrivateState already contains partition id "
               << *partition_index;
    return false;
  }

  //  Validate ranges and expired keys match, if non-empty.
  RangeTracker::InnerMap other_per_key_ranges;
  for (const auto& [key, intervals] : range_tracker) {
    other_per_key_ranges[key] = intervals;
  }
  if (!per_key_ranges_.empty() && per_key_ranges_ != other_per_key_ranges) {
    LOG(ERROR) << "Mismatched per_key_ranges between partitions.";
    return false;
  }
  if (!expired_keys_.empty() &&
      expired_keys_ != range_tracker.GetExpiredKeys()) {
    LOG(ERROR) << "Mismatched expired_keys between partitions.";
    return false;
  }

  // All checks passed, update the state.
  if (per_key_ranges_.empty()) {
    per_key_ranges_ = std::move(other_per_key_ranges);
  }
  if (expired_keys_.empty()) {
    expired_keys_ = range_tracker.GetExpiredKeys();
  }
  symmetric_keys_[*partition_index] = std::string(symmetric_key);
  return true;
}

bool PartitionPrivateState::Merge(const PartitionPrivateState& other) {
  // Validate that no partition ids overlap.
  for (const auto& [id, _] : other.symmetric_keys_) {
    if (symmetric_keys_.contains(id)) {
      LOG(ERROR) << "PartitionPrivateState already contains partition id "
                 << id;
      return false;
    }
  }

  //  Validate ranges and expired keys match, if non-empty.
  if (!per_key_ranges_.empty() && per_key_ranges_ != other.per_key_ranges_) {
    LOG(ERROR) << "Mismatched per_key_ranges between private states.";
    return false;
  }
  if (!expired_keys_.empty() && expired_keys_ != other.expired_keys_) {
    LOG(ERROR) << "Mismatched expired_keys between private states.";
    return false;
  }

  // All checks passed, update the state.
  if (per_key_ranges_.empty()) {
    per_key_ranges_ = other.per_key_ranges_;
  }
  if (expired_keys_.empty()) {
    expired_keys_ = other.expired_keys_;
  }
  symmetric_keys_.insert(other.symmetric_keys_.begin(),
                         other.symmetric_keys_.end());
  return true;
}

std::string PartitionPrivateState::GetSerializedKeys() const {
  FedSqlContainerPartitionKeys proto;
  for (const auto& [id, symmetric_key] : symmetric_keys_) {
    auto* entry = proto.add_keys();
    entry->set_partition_index(id);
    entry->set_symmetric_key(symmetric_key);
  }
  return proto.SerializeAsString();
}

}  // namespace confidential_federated_compute::fed_sql