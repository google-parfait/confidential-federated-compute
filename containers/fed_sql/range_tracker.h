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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_RANGE_TRACKER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_RANGE_TRACKER_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/fed_sql/interval.h"
#include "containers/fed_sql/interval_set.h"
#include "containers/fed_sql/range_tracker.pb.h"

namespace confidential_federated_compute::fed_sql {

// Tracks FedSql specific private state, which consists of ranges of blobs
// visited during execution in FedSql Confidential Transform.
class RangeTracker {
 public:
  RangeTracker() = default;

  // This class is move-only.
  RangeTracker(const RangeTracker&) = delete;
  RangeTracker& operator=(const RangeTracker&) = delete;

  RangeTracker(RangeTracker&&) = default;
  RangeTracker& operator=(RangeTracker&&) = default;

  // Deserializes RangeTracker.
  static absl::StatusOr<RangeTracker> Parse(const RangeTrackerState& state);

  // Deserializes RangeTracker from a string.
  static absl::StatusOr<RangeTracker> Parse(const std::string& data);

  // Serializes the current state.
  RangeTrackerState Serialize() const;

  // Serializes the current state to a string.
  std::string SerializeAsString() const;

  void AddKey(const std::string& key);

  // Tracks [start, end) range as visited for the specified key. This method
  // returns false if this range overlaps with an already visited range.
  bool AddRange(uint64_t start, uint64_t end);

  // Merges the aggregation time window with this RangeTracker
  void MergeAggWindow(Interval<uint64_t> agg_window);

  // Merges the range data with another RangeTracker.
  // The sets of keys are combined, and the ranges are merged together.
  // This method returns false if there are any overlapping ranges.
  bool Merge(const RangeTracker& other);

  // A variant of Merge that merges the underlying data directly.
  bool Merge(const absl::flat_hash_set<std::string>& keys,
             const IntervalSet<uint64_t>& ranges,
             const absl::flat_hash_set<std::string>& expired_keys,
             std::optional<Interval<uint64_t>> agg_window = std::nullopt);

  // Returns the set of keys that are currently being tracked by this
  // RangeTracker.
  const absl::flat_hash_set<std::string>& GetKeys() const { return keys_; }

  // Returns the set of ranges that have been visited by this RangeTracker.
  const IntervalSet<uint64_t>& GetRanges() const { return ranges_; }

  // Sets the index of the partition that this RangeTracker is tracking.
  void SetPartitionIndex(uint64_t index) { partition_index_ = index; }

  // Returns the index of the partition that this RangeTracker is tracking.
  std::optional<uint64_t> GetPartitionIndex() const { return partition_index_; }

  // Gets the expired keys for this RangeTracker.
  const absl::flat_hash_set<std::string>& GetExpiredKeys() const {
    return expired_keys_;
  }
  // Sets the expired keys for this RangeTracker.
  void SetExpiredKeys(const absl::flat_hash_set<std::string>& expired_keys) {
    expired_keys_ = expired_keys;
  }

  // Returns the aggregation time window, if set.
  std::optional<Interval<uint64_t>> GetAggregationWindow() const {
    return agg_window_;
  }

 private:
  // Set of keys that are currently being tracked by this RangeTracker.
  absl::flat_hash_set<std::string> keys_;

  // Stores visited ranges of blobs. These are tracked across all keys.
  IntervalSet<uint64_t> ranges_;

  // The index of the partition that this RangeTracker is tracking.
  std::optional<uint64_t> partition_index_;

  // Keys that have already expired and must be removed from the budget.
  absl::flat_hash_set<std::string> expired_keys_;

  // The aggregation time window.
  std::optional<Interval<uint64_t>> agg_window_;
};

// Serializes RangeTracker and bundles it to a blob, and returns a combined
// blob.
std::string BundleRangeTracker(std::string blob,
                               const RangeTracker& range_tracker);

// Extracts serialized RangeTracker from a blob and replaces the blob in place.
absl::StatusOr<RangeTracker> UnbundleRangeTracker(std::string& blob);

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_RANGE_TRACKER_H_