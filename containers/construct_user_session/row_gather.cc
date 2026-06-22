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
#include "absl/types/span.h"
#include "containers/common/row_set.h"
#include "containers/construct_user_session/ingestion.h"
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

// Gathers values for a single column across all rows in `group`. Returns
// without inserting into `result` if the column is absent from every input.
//
// IMPORTANT: `inputs` must outlive the returned tensors in `result`.
template <typename T>
void GatherColumn(DataType dtype, absl::string_view col_name,
                  absl::Span<const RowLocation> group,
                  absl::Span<const Checkpoint> inputs,
                  absl::flat_hash_map<std::string, Tensor>& result) {
  std::vector<T> values;
  values.reserve(group.size());

  // Caches the most recently used input tensor to avoid repeated lookups.
  // We require that `group` is sorted primarily by `group_key` and secondarily
  // by `input_index`, so consecutive rows will often share the same input
  // tensor.
  uint32_t cached_input_index = UINT32_MAX;
  absl::Span<const T> cached_span;
  bool column_present = false;

  for (const auto& loc : group) {
    DCHECK_LT(loc.input_index, inputs.size());
    if (loc.input_index != cached_input_index) {
      cached_input_index = loc.input_index;
      auto it = inputs[cached_input_index].column_tensors().find(col_name);
      if (it != inputs[cached_input_index].column_tensors().end()) {
        cached_span = it->second.AsSpan<T>();
        column_present = true;
      } else {
        cached_span = {};
      }
    }
    if (cached_span.empty()) continue;
    DCHECK_LT(loc.row_index, cached_span.size());
    values.push_back(cached_span[loc.row_index]);
  }

  if (!column_present)
    return;  // column absent from every input — don't add to result.

  if (values.size() < group.size()) {
    LOG(WARNING) << "Tensor '" << col_name << "' present in " << values.size()
                 << " of " << group.size() << " rows for privacy ID group";
  }

  // Uses VectorData<T> to avoid copying the underlying data. For DT_STRING
  // (T = string_view), the resulting tensor holds non-owning views into the
  // string data owned by the source tensors in `inputs`.
  const int64_t num_values = static_cast<int64_t>(values.size());
  auto data = std::make_unique<VectorData<T>>(std::move(values));
  auto tensor = Tensor::Create(dtype, TensorShape{num_values}, std::move(data),
                               std::string(col_name));
  CHECK_OK(tensor.status());
  result.emplace(std::string(col_name), *std::move(tensor));
}

}  // namespace

absl::flat_hash_map<std::string, Tensor> GatherSurvivingRows(
    absl::Span<const RowLocation> group, absl::Span<const Checkpoint> inputs,
    const absl::flat_hash_map<std::string, DataType>& column_dtypes) {
  absl::flat_hash_map<std::string, Tensor> result;

  // Gather surviving rows for each column.
  for (const auto& [col_name, dtype] : column_dtypes) {
    DTYPE_CASES(dtype, T,
                GatherColumn<T>(dtype, col_name, group, inputs, result));
  }

  return result;
}

}  // namespace confidential_federated_compute::construct_user_session
