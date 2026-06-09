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

#include "containers/construct_user_session/ingestion.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "containers/common/checkpoint_utils.h"
#include "containers/common/row_set.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute::construct_user_session {

namespace {
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivacyIdColumnName;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::Tensor;

// Parses event time strings into a vector of absl::Time values.
// Malformed timestamps are logged and replaced with absl::InfinitePast().
std::vector<absl::Time> ParseEventTimeStrings(
    const std::vector<std::string>& raw_strings) {
  std::vector<absl::Time> times;
  times.reserve(raw_strings.size());
  for (const std::string& s : raw_strings) {
    absl::Time t;
    std::string err;
    if (!absl::ParseTime(absl::RFC3339_full, s, &t, &err)) {
      LOG(WARNING) << "Failed to parse event time '" << s << "': " << err
                   << ". Treating as absl::InfinitePast().";
      t = absl::InfinitePast();
    }
    times.push_back(t);
  }
  return times;
}
}  // namespace

absl::StatusOr<DeserializedCheckpoint> DeserializeCheckpoint(
    CheckpointParser* checkpoint, absl::string_view on_device_query_name) {
  if (checkpoint == nullptr) {
    return absl::InvalidArgumentError("checkpoint must not be null.");
  }

  // Extract system tensors via the shared helpers (which handle all
  // dtype and shape validation internally).
  FCP_ASSIGN_OR_RETURN(std::string privacy_id, GetPrivacyId(*checkpoint));
  FCP_ASSIGN_OR_RETURN(std::vector<std::string> raw_event_times,
                       GetEventTime(*checkpoint, on_device_query_name));
  std::vector<absl::Time> event_times = ParseEventTimeStrings(raw_event_times);
  const size_t num_rows = event_times.size();

  // Load all tensors and erase the two system tensor keys that have
  // already been processed.
  FCP_ASSIGN_OR_RETURN(auto all_tensors, checkpoint->LoadAllTensors());
  all_tensors.erase(std::string(kPrivacyIdColumnName));
  all_tensors.erase(
      absl::StrCat(on_device_query_name, "/", kEventTimeColumnName));

  // Validate remaining data tensors and build the output map.
  absl::flat_hash_map<std::string, Tensor> data_tensors;
  data_tensors.reserve(all_tensors.size());
  for (auto& [name, tensor] : all_tensors) {
    if (tensor.shape().dim_sizes().size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Data tensor `%s` must have one dimension.", name));
    }
    if (tensor.num_elements() != num_rows) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Data tensor `%s` has %d rows, expected %d matching "
                          "event time tensor.",
                          name, tensor.num_elements(), num_rows));
    }
    data_tensors.emplace(name, std::move(tensor));
  }

  return DeserializedCheckpoint{
      .privacy_id = std::move(privacy_id),
      .event_times = std::move(event_times),
      .data_tensors = std::move(data_tensors),
  };
}

std::vector<RowLocation> FilterForSessionWindow(
    absl::Span<const absl::Time> event_times, uint64_t group_key,
    uint32_t input_index, absl::Time window_start, absl::Time window_end) {
  std::vector<RowLocation> row_locations;
  row_locations.reserve(event_times.size());

  for (size_t row_index = 0; row_index < event_times.size(); ++row_index) {
    const absl::Time t = event_times[row_index];
    // Half-open interval: [window_start, window_end).
    // Sentinel values (absl::InfinitePast()) are naturally excluded since they
    // fall outside any valid session window.
    if (t >= window_start && t < window_end) {
      row_locations.push_back(
          RowLocation{.group_key = group_key,
                      .input_index = input_index,
                      .row_index = static_cast<uint32_t>(row_index)});
    }
  }

  return row_locations;
}

}  // namespace confidential_federated_compute::construct_user_session
