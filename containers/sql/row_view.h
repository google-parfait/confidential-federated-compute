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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_ROW_VIEW_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_ROW_VIEW_H_

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "fcp/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sql {

// A non-owning view of a single row, which is composed of elements from
// multiple columns (Tensors).
class RowView {
 public:
  static absl::StatusOr<RowView> Create(
      absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
      uint32_t row_index);

  tensorflow_federated::aggregation::DataType GetColumnType(
      int column_index) const {
    return columns_[column_index].dtype();
  }

  // Returns the value of a cell in the row.
  template <typename T>
  T GetValue(int column_index) const {
    const auto& column = columns_[column_index];
    // This will CHECK-fail if T does not match the column's dtype.
    return column.AsSpan<T>().at(row_index_);
  }

  size_t GetColumnCount() const { return columns_.size(); }

 private:
  RowView(absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
          uint32_t row_index)
      : columns_(columns), row_index_(row_index) {};

  absl::Span<const tensorflow_federated::aggregation::Tensor> columns_;
  uint32_t row_index_;
};

}  // namespace confidential_federated_compute::sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_ROW_VIEW_H_
