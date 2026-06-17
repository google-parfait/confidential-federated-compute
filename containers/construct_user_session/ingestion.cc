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
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "containers/common/row_set.h"
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
}  // namespace

absl::string_view Checkpoint::privacy_id() const {
  return privacy_id_tensor_.AsScalar<absl::string_view>();
}

Tensor Checkpoint::take_privacy_id_tensor() {
  return std::move(privacy_id_tensor_);
}

const absl::flat_hash_map<std::string, Tensor>& Checkpoint::column_tensors()
    const {
  return column_tensors_;
}

absl::StatusOr<Checkpoint> Checkpoint::Create(
    CheckpointParser& checkpoint, absl::string_view on_device_query_name) {
  ABSL_ASSIGN_OR_RETURN(auto all_tensors, checkpoint.LoadAllTensors());

  // Extract and validate the privacy ID tensor.
  auto privacy_id_node = all_tensors.extract(std::string(kPrivacyIdColumnName));
  if (privacy_id_node.empty()) {
    return absl::NotFoundError(absl::StrFormat(
        "Tensor `%s` not found in checkpoint", kPrivacyIdColumnName));
  }
  Tensor& privacy_id = privacy_id_node.mapped();
  if (privacy_id.dtype() != DataType::DT_STRING) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must be a string tensor", kPrivacyIdColumnName));
  }
  if (!privacy_id.is_scalar()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`%s` tensor must be a scalar", kPrivacyIdColumnName));
  }

  // Validate the event time tensor in-place (it stays in the map).
  const std::string event_time_name =
      absl::StrCat(on_device_query_name, "/", kEventTimeColumnName);
  auto event_time_it = all_tensors.find(event_time_name);
  if (event_time_it == all_tensors.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Tensor `%s` not found in checkpoint", event_time_name));
  }
  const Tensor& event_time = event_time_it->second;
  if (event_time.dtype() != DataType::DT_STRING) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must be a string tensor", event_time_name));
  }
  if (event_time.shape().dim_sizes().size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must have one dimension", event_time_name));
  }
  const size_t num_rows = event_time.num_elements();

  // Validate remaining column tensors.
  for (const auto& [name, tensor] : all_tensors) {
    // Event time tensor has already been validated.
    if (name == event_time_name) {
      continue;
    }
    if (tensor.shape().dim_sizes().size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Column tensor `%s` must have one dimension.", name));
    }
    if (tensor.num_elements() != num_rows) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Column tensor `%s` has %d rows, expected %d matching "
          "event time tensor.",
          name, tensor.num_elements(), num_rows));
    }
  }

  return Checkpoint(std::move(privacy_id), std::move(all_tensors));
}

Checkpoint::Checkpoint(Tensor privacy_id_tensor,
                       absl::flat_hash_map<std::string, Tensor> column_tensors)
    : privacy_id_tensor_(std::move(privacy_id_tensor)),
      column_tensors_(std::move(column_tensors)) {}

std::vector<RowLocation> FilterForSessionWindow(const Tensor& event_times,
                                                uint64_t group_key,
                                                uint32_t input_index,
                                                absl::Time window_start,
                                                absl::Time window_end) {
  const size_t num_rows = event_times.num_elements();
  std::vector<RowLocation> row_locations;
  row_locations.reserve(num_rows);

  absl::Span<const absl::string_view> event_times_span =
      event_times.AsSpan<absl::string_view>();
  for (size_t row_index = 0; row_index < num_rows; ++row_index) {
    absl::Time t;
    std::string err;
    if (!absl::ParseTime(absl::RFC3339_full, event_times_span[row_index], &t,
                         &err)) {
      LOG(WARNING) << "Failed to parse event time: " << err
                   << ". Excluding from session.";
      continue;
    }
    // Half-open interval: [window_start, window_end).
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
