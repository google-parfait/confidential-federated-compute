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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_INGESTION_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_INGESTION_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute::construct_user_session {

// Represents the extracted "system columns" (privacy ID and event time) and
// data tensors from a client checkpoint.
struct DeserializedCheckpoint {
  // The precise Privacy ID extracted from the scalar
  // `confidential_compute_privacy_id` string tensor.
  std::string privacy_id;

  // Parsed event times from
  // `<on_device_query_name>/confidential_compute_event_time`. Malformed
  // timestamps are stored as absl::InfinitePast().
  std::vector<absl::Time> event_times;

  // All data tensors found in the checkpoint keyed by their tensor name. Each
  // tensor is 1-dimensional with the same row count as `event_times`.
  absl::flat_hash_map<std::string, tensorflow_federated::aggregation::Tensor>
      data_tensors;
};

// Deserializes a client checkpoint into its constituent parts using dynamic
// tensor discovery.
//
// This function:
// (1) Loads all tensors from the checkpoint.
// (2) Extracts the scalar `confidential_compute_privacy_id` tensor.
// (3) Extracts the 1D `<on_device_query_name>/confidential_compute_event_time`
//     string tensor and parses each ISO-8601 event time string into an
//     absl::Time. Malformed timestamps are stored as absl::InfinitePast()
//     sentinels.
// (4) stores all remaining tensors in the `data_tensors` map (name → Tensor),
// (5) validates that the privacy ID tensor is scalar, that the event time
//     tensor is 1-dimensional, that all data tensors are 1-dimensional, and
//     that all data tensors have the same row count as event_times.size().
absl::StatusOr<DeserializedCheckpoint> DeserializeCheckpoint(
    tensorflow_federated::aggregation::CheckpointParser* checkpoint,
    absl::string_view on_device_query_name);

}  // namespace confidential_federated_compute::construct_user_session

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_INGESTION_H_
