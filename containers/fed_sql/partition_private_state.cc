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

#include <variant>

#include "absl/functional/overload.h"
#include "absl/log/log.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"

namespace confidential_federated_compute::fed_sql {

using ::fcp::confidentialcompute::
    FedSqlContainerPartitionedOutputFinalizedState;

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

  state.expired_keys_ = absl::flat_hash_set<std::string>(
      proto.expired_keys().begin(), proto.expired_keys().end());

  if (proto.values_size() % 2 != 0) {
    return absl::InvalidArgumentError(
        "Unexpected number of values in serialized PartitionPrivateState.");
  }
  std::vector<Interval<uint64_t>> intervals;
  intervals.reserve(proto.values_size() / 2);
  for (int i = 0; i < proto.values_size(); i += 2) {
    intervals.emplace_back(proto.values(i), proto.values(i + 1));
  }
  if (!state.ranges_.Assign(intervals.begin(), intervals.end())) {
    return absl::InvalidArgumentError(
        "Unexpected order of intervals in serialized PartitionPrivateState.");
  }

  if (proto.has_start_time() != proto.has_end_time()) {
    return absl::InternalError(
        "PartitionPrivateState proto must have either both start_time and "
        "end_time or neither.");
  }

  bool has_keys = !proto.keys().empty();
  bool has_agg_window = proto.has_start_time();
  if (has_keys && has_agg_window) {
    return absl::InvalidArgumentError(
        "PartitionPrivateState proto must not have both keys and an "
        "aggregation window.");
  }

  if (has_keys) {
    state.keys_or_agg_window_ = absl::flat_hash_set<std::string>(
        proto.keys().begin(), proto.keys().end());
  } else if (has_agg_window) {
    state.keys_or_agg_window_ = Interval<uint64_t>(proto.start_time().seconds(),
                                                   proto.end_time().seconds());
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
  std::visit(absl::Overload{
                 [](std::monostate) {},
                 [&](const absl::flat_hash_set<std::string>& keys) {
                   for (const auto& key : keys) {
                     proto.add_keys(key);
                   }
                 },
                 [&](const Interval<uint64_t>& window) {
                   proto.mutable_start_time()->set_seconds(window.start());
                   proto.mutable_end_time()->set_seconds(window.end());
                 },
             },
             keys_or_agg_window_);
  for (const auto& interval : ranges_) {
    proto.add_values(interval.start());
    proto.add_values(interval.end());
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

  // Validate ranges, keys/agg_window, and expired keys match, if non-empty.
  if (!ranges_.empty() && ranges_ != range_tracker.GetRanges()) {
    LOG(ERROR) << "Mismatched ranges between partitions.";
    return false;
  }
  if (!std::holds_alternative<std::monostate>(keys_or_agg_window_) &&
      keys_or_agg_window_ != range_tracker.GetKeysOrAggWindow()) {
    LOG(ERROR) << "Mismatched keys or aggregation window between partitions.";
    return false;
  }
  if (!expired_keys_.empty() &&
      expired_keys_ != range_tracker.GetExpiredKeys()) {
    LOG(ERROR) << "Mismatched expired_keys between partitions.";
    return false;
  }

  // All checks passed, update the state.
  if (ranges_.empty()) {
    ranges_ = range_tracker.GetRanges();
  }
  if (std::holds_alternative<std::monostate>(keys_or_agg_window_)) {
    keys_or_agg_window_ = range_tracker.GetKeysOrAggWindow();
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

  // Validate ranges, keys/agg_window, and expired keys match, if non-empty.
  if (!ranges_.empty() && ranges_ != other.ranges_) {
    LOG(ERROR) << "Mismatched ranges between private states.";
    return false;
  }
  if (!std::holds_alternative<std::monostate>(keys_or_agg_window_) &&
      keys_or_agg_window_ != other.keys_or_agg_window_) {
    LOG(ERROR)
        << "Mismatched keys or aggregation window between private states.";
    return false;
  }
  if (!expired_keys_.empty() && expired_keys_ != other.expired_keys_) {
    LOG(ERROR) << "Mismatched expired_keys between private states.";
    return false;
  }

  // All checks passed, update the state.
  if (ranges_.empty()) {
    ranges_ = other.ranges_;
  }
  if (std::holds_alternative<std::monostate>(keys_or_agg_window_)) {
    keys_or_agg_window_ = other.keys_or_agg_window_;
  }
  if (expired_keys_.empty()) {
    expired_keys_ = other.expired_keys_;
  }
  symmetric_keys_.insert(other.symmetric_keys_.begin(),
                         other.symmetric_keys_.end());
  return true;
}

}  // namespace confidential_federated_compute::fed_sql