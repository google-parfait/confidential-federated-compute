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
#include "containers/common/io/tabular/row_view.h"

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/message.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute {

absl::StatusOr<RowView> RowView::CreateFromTensors(
    absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
    uint32_t row_index) {
  ABSL_ASSIGN_OR_RETURN(TensorRowView tensor_row_view,
                        TensorRowView::Create(columns, row_index));
  return RowView(std::move(tensor_row_view));
}

absl::StatusOr<RowView> RowView::CreateFromMessage(
    const google::protobuf::Message* message,
    absl::Span<const tensorflow_federated::aggregation::Tensor> system_columns,
    uint32_t row_index, const MessageColumnSchema* schema) {
  return RowView(MessageRowView(message, system_columns, row_index, schema));
}

absl::StatusOr<RowView::TensorRowView> RowView::TensorRowView::Create(
    absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
    uint32_t row_index) {
  if (columns.empty()) {
    return absl::InvalidArgumentError("No columns provided.");
  }
  if (columns[0].shape().dim_sizes().empty()) {
    return absl::InvalidArgumentError("Column has no rows.");
  }
  if (row_index >= columns[0].shape().dim_sizes()[0]) {
    return absl::InvalidArgumentError("Row index is out of bounds.");
  }

  return TensorRowView(columns, row_index);
}

RowView::MessageRowView::MessageRowView(
    const google::protobuf::Message* message,
    absl::Span<const tensorflow_federated::aggregation::Tensor> system_columns,
    uint32_t row_index, const MessageColumnSchema* schema)
    : message_(message),
      system_columns_(system_columns),
      row_index_(row_index),
      schema_(schema) {}

tensorflow_federated::aggregation::DataType
RowView::MessageRowView::GetColumnType(int column_index) const {
  return (*schema_)[column_index].column_type;
}

size_t RowView::MessageRowView::GetColumnCount() const {
  return schema_->size();
}

}  // namespace confidential_federated_compute
