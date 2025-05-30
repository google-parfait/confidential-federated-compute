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
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/fed_sql/interval_set.h"
#include "containers/fed_sql/range_tracker.pb.h"

namespace confidential_federated_compute::fed_sql {

// Signature written in the beginning of the bundled blob that
// contains serialized RangeTracker state.
inline constexpr const char kRangeTrackerBundleSignature[] = "RTv1";

// Tracks FedSql specific private state, which consists of ranges of blobs
// visited during execution in FedSql Confidential Transform.
class RangeTracker {
 public:
  using InnerMap = absl::flat_hash_map<std::string, IntervalSet<uint64_t>>;
  using const_iterator = typename InnerMap::const_iterator;
  using value_type = typename InnerMap::value_type;

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

  // Tracks [start, end) range as visited for the specified key. This method
  // returns false if this range overlaps with an already visited range.
  bool AddRange(const std::string& key, uint64_t start, uint64_t end);

  // Merges the range data with another RangeTracker.
  // This method returns false if there are any overlapping ranges.
  bool Merge(const RangeTracker& other);

  // Iteration support.
  const_iterator begin() const { return per_key_ranges_.begin(); }
  const_iterator end() const { return per_key_ranges_.end(); }

 private:
  // Stores visited ranges of blobs organized by key_id of encryption key
  // used to encrypt a blob (the same key_id that is found in BlobHeader).
  InnerMap per_key_ranges_;
};

// Serializes RangeTracker and bundles it to a blob, and returns a combined
// blob.
std::string BundleRangeTracker(const std::string& blob,
                               const RangeTracker& range_tracker);

// Extracts serialized RangeTracker from a blob and replaces the blob in place.
absl::StatusOr<RangeTracker> UnbundleRangeTracker(std::string& blob);

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_RANGE_TRACKER_H_