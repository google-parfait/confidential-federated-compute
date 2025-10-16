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
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "fcp/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sql {

// A non-owning view of a single row of data, abstracting the underlying
// storage mechanism (e.g., Tensors, Messages) via absl::variant.
class RowView {
 public:
  // Creates a RowView from a span of columns and a row index.
  // A RowView created this way will provide access to the elements of the row
  // by index, in the order of the `columns` span.
  static absl::StatusOr<RowView> CreateFromTensors(
      absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
      uint32_t row_index);

  // Returns the data type of a column.
  tensorflow_federated::aggregation::DataType GetColumnType(
      int column_index) const {
    return absl::visit(
        [column_index](const auto& view) {
          return view.GetColumnType(column_index);
        },
        row_view_variant_);
  }

  // Returns the value of an element in the row.
  template <typename T>
  T GetValue(int column_index) const {
    return absl::visit(
        [column_index](const auto& view) {
          return view.template GetValue<T>(column_index);
        },
        row_view_variant_);
  }

  // Returns the number of columns in the row.
  size_t GetColumnCount() const {
    return absl::visit([](const auto& view) { return view.GetColumnCount(); },
                       row_view_variant_);
  }

 private:
  // Type trait to check if a type T conforms to the RowView interface.
  template <typename T, typename = void>
  struct has_row_view_interface : std::false_type {};

  template <typename T>
  struct has_row_view_interface<
      T, std::void_t<
             decltype(std::declval<const T&>().GetColumnType(0)),
             decltype(std::declval<const T&>().template GetValue<int32_t>(0)),
             decltype(std::declval<const T&>().GetColumnCount())>>
      : std::true_type {};

  // A RowView backed by Tensors.
  class TensorRowView {
   public:
    static absl::StatusOr<TensorRowView> Create(
        absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
        uint32_t row_index);

    tensorflow_federated::aggregation::DataType GetColumnType(
        int column_index) const {
      return columns_[column_index].dtype();
    }

    template <typename T>
    T GetValue(int column_index) const {
      const auto& column = columns_[column_index];
      // This will CHECK-fail if T does not match the column's dtype.
      return column.AsSpan<T>().at(row_index_);
    }

    size_t GetColumnCount() const { return columns_.size(); }

   private:
    TensorRowView(
        absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
        uint32_t row_index)
        : columns_(columns), row_index_(row_index) {};

    absl::Span<const tensorflow_federated::aggregation::Tensor> columns_;
    uint32_t row_index_;
  };

  static_assert(has_row_view_interface<TensorRowView>::value,
                "TensorRowView does not conform to the RowView interface.");

  //  TODO: add a MessageRowView.
  using RowViewVariant = absl::variant<TensorRowView>;

  explicit RowView(RowViewVariant row_view_variant)
      : row_view_variant_(std::move(row_view_variant)) {}

  RowViewVariant row_view_variant_;
};

}  // namespace confidential_federated_compute::sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_ROW_VIEW_H_
