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

#include "containers/construct_user_session/row_gather.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "containers/construct_user_session/checkpoint.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/vector_data.h"

namespace confidential_federated_compute::construct_user_session {

using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::string_view;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::tensorflow_federated::aggregation::VectorData;

// TODO: Remove these once DTYPE_CASES is updated.

// Aliases required by the DTYPE_CASES macro, which references DT_INVALID and
// internal::TypeTraits<T> without namespace qualification.
constexpr auto DT_INVALID = ::tensorflow_federated::aggregation::DT_INVALID;

namespace internal {
template <typename T>
using TypeTraits = ::tensorflow_federated::aggregation::internal::TypeTraits<T>;
}  // namespace internal

namespace {

// Returns the indices of rows whose event times fall within
// [window_start, window_end). Malformed timestamps are excluded.
std::vector<uint32_t> FilterRowIndices(
    absl::Span<const absl::string_view> event_times, absl::Time window_start,
    absl::Time window_end) {
  std::vector<uint32_t> indices;
  indices.reserve(event_times.size());
  for (size_t i = 0; i < event_times.size(); ++i) {
    absl::Time t;
    std::string err;
    if (!absl::ParseTime(absl::RFC3339_full, event_times[i], &t, &err)) {
      LOG(WARNING) << "Failed to parse event time: " << err
                   << ". Excluding from session.";
      continue;
    }
    if (t >= window_start && t < window_end) {
      indices.push_back(static_cast<uint32_t>(i));
    }
  }
  return indices;
}

// Gathers values at `row_indices` for tensor `tensor_name` from a single
// checkpoint and appends them to `values`. If the tensor is not present
// in the checkpoint, `values` is unchanged.
template <typename T>
void AppendTensorValues(absl::string_view tensor_name,
                        const Checkpoint& checkpoint,
                        absl::Span<const uint32_t> row_indices,
                        std::vector<T>& values) {
  auto it = checkpoint.column_tensors().find(tensor_name);
  if (it == checkpoint.column_tensors().end()) {
    LOG(WARNING) << "Tensor not found in checkpoint.";
    return;
  }
  absl::Span<const T> span = it->second.AsSpan<T>();
  // Indices are naturally ordered, so checking the last one is sufficient.
  if (!row_indices.empty()) {
    DCHECK_LT(row_indices.back(), span.size());
  }
  for (uint32_t idx : row_indices) {
    values.push_back(span[idx]);
  }
}

// Gathers values for a single tensor across all checkpoints using
// pre-filtered row indices. Inserts the result into `result` if any values
// were gathered.
template <typename T>
void GatherTensor(DataType dtype, absl::string_view tensor_name,
                  absl::Span<const Checkpoint> checkpoints,
                  absl::Span<const std::vector<uint32_t>> filtered_indices,
                  size_t total_surviving_rows,
                  absl::flat_hash_map<std::string, Tensor>& result) {
  std::vector<T> values;
  values.reserve(total_surviving_rows);

  for (size_t i = 0; i < checkpoints.size(); ++i) {
    AppendTensorValues<T>(tensor_name, checkpoints[i], filtered_indices[i],
                          values);
  }

  // Tensor absent from every input, or all rows were filtered out.
  if (values.empty()) return;

  // Uses VectorData<T> to avoid copying the underlying data. For DT_STRING
  // (T = string_view), the resulting tensor holds non-owning views into the
  // string data owned by the source tensors in `inputs`.
  const int64_t num_values = static_cast<int64_t>(values.size());
  auto data = std::make_unique<VectorData<T>>(std::move(values));
  auto tensor = Tensor::Create(dtype, TensorShape{num_values}, std::move(data),
                               std::string(tensor_name));
  CHECK_OK(tensor.status());
  result.emplace(std::string(tensor_name), *std::move(tensor));
}

}  // namespace

absl::flat_hash_map<std::string, Tensor> GatherSessionRows(
    absl::Span<const Checkpoint> checkpoints,
    absl::string_view event_time_tensor_name, absl::Time window_start,
    absl::Time window_end,
    const absl::flat_hash_map<std::string, DataType>& column_dtypes) {
  // Get the indices of rows within the session window for each checkpoint.
  std::vector<std::vector<uint32_t>> filtered_indices;
  filtered_indices.reserve(checkpoints.size());
  for (const auto& checkpoint : checkpoints) {
    const Tensor& event_time =
        checkpoint.column_tensors().at(event_time_tensor_name);
    filtered_indices.push_back(FilterRowIndices(
        event_time.AsSpan<absl::string_view>(), window_start, window_end));
  }

  // Compute the total number of surviving rows across all checkpoints.
  size_t total_surviving_rows = 0;
  for (const auto& indices : filtered_indices) {
    total_surviving_rows += indices.size();
  }

  // Gather surviving values for each tensor.
  absl::flat_hash_map<std::string, Tensor> result;
  for (const auto& [tensor_name, dtype] : column_dtypes) {
    DTYPE_CASES(
        dtype, T,
        GatherTensor<T>(dtype, tensor_name, checkpoints, filtered_indices,
                        total_surviving_rows, result));
  }

  return result;
}

}  // namespace confidential_federated_compute::construct_user_session
