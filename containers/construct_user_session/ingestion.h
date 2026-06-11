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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "containers/common/row_set.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute::construct_user_session {

// Represents the extracted "system columns" (privacy ID and event time) and
// data tensors from a client checkpoint.
struct DeserializedCheckpoint {
  // The precise Privacy ID extracted from the scalar
  // `confidential_compute_privacy_id` string tensor.
  std::string privacy_id;

  // Raw event time tensor from
  // `<on_device_query_name>/confidential_compute_event_time`.
  // Each element is expected to have the format YYYY-MM-DDTHH:MM:SS±HH:MM.
  tensorflow_federated::aggregation::Tensor event_times;

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
//     string tensor and stores it as-is in `event_times`.
// (4) Stores all remaining tensors in the `data_tensors` map (name → Tensor),
// (5) Validates that the privacy ID tensor is scalar, that the event time
//     tensor is 1-dimensional, that all data tensors are 1-dimensional, and
//     that all data tensors have the same row count as event_times.
absl::StatusOr<DeserializedCheckpoint> DeserializeCheckpoint(
    tensorflow_federated::aggregation::CheckpointParser& checkpoint,
    absl::string_view on_device_query_name);

// Filters event times against the session window [window_start, window_end)
// and produces RowLocation metadata for surviving rows.
//
// For each event time string in `event_times`, the string is parsed to
// absl::Time. If the parsed time satisfies `window_start <= t < window_end`,
// a RowLocation is created with the given `group_key`, `input_index`, and the
// row's index within `event_times`.
// Malformed timestamps that fail to parse are logged and excluded.
std::vector<RowLocation> FilterForSessionWindow(
    const tensorflow_federated::aggregation::Tensor& event_times,
    uint64_t group_key, uint32_t input_index, absl::Time window_start,
    absl::Time window_end);

}  // namespace confidential_federated_compute::construct_user_session

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_INGESTION_H_
