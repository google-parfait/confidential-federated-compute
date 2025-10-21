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
#include "containers/sql/row_view.h"

#include "absl/status/status.h"
#include "fcp/base/monitoring.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/message.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sql {

absl::StatusOr<RowView> RowView::CreateFromTensors(
    absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
    uint32_t row_index) {
  FCP_ASSIGN_OR_RETURN(TensorRowView tensor_row_view,
                       TensorRowView::Create(columns, row_index));
  return RowView(std::move(tensor_row_view));
}

absl::StatusOr<RowView> RowView::CreateFromMessage(
    const google::protobuf::Message* message,
    absl::Span<const tensorflow_federated::aggregation::Tensor> system_columns,
    uint32_t row_index) {
  return RowView(MessageRowView(message, system_columns, row_index));
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
    uint32_t row_index)
    : message_(message),
      reflection_(message->GetReflection()),
      descriptor_(message->GetDescriptor()),
      system_columns_(system_columns),
      row_index_(row_index) {}

size_t RowView::MessageRowView::GetSystemColumnIndex(int column_index) const {
  return column_index - descriptor_->field_count();
}

tensorflow_federated::aggregation::DataType
RowView::MessageRowView::GetMessageColumnType(int column_index) const {
  const google::protobuf::FieldDescriptor* field =
      descriptor_->field(column_index);
  switch (field->cpp_type()) {
    case google::protobuf::FieldDescriptor::CPPTYPE_INT32:
      return tensorflow_federated::aggregation::DataType::DT_INT32;
    case google::protobuf::FieldDescriptor::CPPTYPE_INT64:
      return tensorflow_federated::aggregation::DataType::DT_INT64;
    case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      return tensorflow_federated::aggregation::DataType::DT_FLOAT;
    case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      return tensorflow_federated::aggregation::DataType::DT_DOUBLE;
    case google::protobuf::FieldDescriptor::CPPTYPE_STRING:
      return tensorflow_federated::aggregation::DataType::DT_STRING;
    default:
      FCP_LOG(FATAL) << "Unsupported column type " << field->cpp_type_name();
  }
}

tensorflow_federated::aggregation::DataType
RowView::MessageRowView::GetColumnType(int column_index) const {
  if (column_index < descriptor_->field_count()) {
    return GetMessageColumnType(column_index);
  }
  return system_columns_[GetSystemColumnIndex(column_index)].dtype();
}

size_t RowView::MessageRowView::GetColumnCount() const {
  return descriptor_->field_count() + system_columns_.size();
}
}  // namespace confidential_federated_compute::sql
