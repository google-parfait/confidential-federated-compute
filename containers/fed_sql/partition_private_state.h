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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_PARTITION_PRIVATE_STATE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_PARTITION_PRIVATE_STATE_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "containers/fed_sql/partition_private_state.pb.h"
#include "containers/fed_sql/range_tracker.h"

namespace confidential_federated_compute::fed_sql {

// Tracks output partition private state such as per-partition symmetric
// keys and the range tracker ranges. This state is passed between containers
// when merging the private states across partitions.
class PartitionPrivateState {
 public:
  PartitionPrivateState() = default;

  // This class is move-only.
  PartitionPrivateState(const PartitionPrivateState&) = delete;
  PartitionPrivateState& operator=(const PartitionPrivateState&) = delete;

  PartitionPrivateState(PartitionPrivateState&&) = default;
  PartitionPrivateState& operator=(PartitionPrivateState&&) = default;

  // Deserializes PartitionPrivateState.
  static absl::StatusOr<PartitionPrivateState> Parse(
      const PartitionPrivateStateProto& state);

  // Deserializes PartitionPrivateState from a string.
  static absl::StatusOr<PartitionPrivateState> Parse(const std::string& data);

  // Serializes the current state.
  PartitionPrivateStateProto Serialize() const;

  // Serializes the current state to a string.
  std::string SerializeAsString() const;

  // Adds a new partition's private state.
  bool AddPartition(const RangeTracker& range_tracker,
                    absl::string_view symmetric_key);

  // Merges another PartitionPrivateState into this one.
  bool Merge(const PartitionPrivateState& state);

 private:
  // Symmetric keys used to encrypt each partition, key-ed by the partition id.
  absl::flat_hash_map<uint64_t, std::string> symmetric_keys_;
  // KMS keys that have already expired and must be removed from the budget.
  absl::flat_hash_set<std::string> expired_keys_;
  // Per key ranges as tracked by the range tracker.
  RangeTracker::InnerMap per_key_ranges_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_PARTITION_PRIVATE_STATE_H_