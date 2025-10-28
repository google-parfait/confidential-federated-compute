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

#include "containers/sql/input.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "containers/sql/row_view.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace confidential_federated_compute::sql {
namespace {

using ::fcp::confidentialcompute::BlobHeader;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorData;
using ::tensorflow_federated::aggregation::TensorShape;

absl::Status ValidateMessageRows(
    absl::Span<const std::unique_ptr<google::protobuf::Message>> messages,
    absl::Span<const Tensor> system_columns) {
  if (messages.empty()) {
    return absl::InvalidArgumentError("No rows provided.");
  }
  const auto& first_row = messages[0];
  const google::protobuf::Descriptor* first_descriptor =
      first_row->GetDescriptor();
  for (const auto& message : messages) {
    if (message->GetDescriptor() != first_descriptor) {
      return absl::InvalidArgumentError(
          "All messages in a table must have the same proto type.");
    }
  }
  for (const auto& system_column : system_columns) {
    if (system_column.shape().dim_sizes().size() != 1) {
      return absl::InvalidArgumentError(
          "System columns must have a single dimension.");
    }
    if (system_column.shape().dim_sizes()[0] != messages.size()) {
      return absl::InvalidArgumentError(
          "System columns must have the same number of rows as the table.");
    }
  }
  return absl::OkStatus();
}

template <typename T>
std::unique_ptr<TensorData> CreateVectorTensorData(
    size_t num_rows, absl::Span<const RowView> row_views, size_t column_index) {
  auto builder = std::make_unique<MutableVectorData<T>>();
  builder->reserve(num_rows);
  for (const auto& row_view : row_views) {
    builder->push_back(row_view.GetValue<T>(column_index));
  }
  return builder;
}

}  // namespace
Input::Input(ContentsVariant contents,
             fcp::confidentialcompute::BlobHeader blob_header,
             std::vector<std::string> column_names)
    : contents_(std::move(contents)),
      blob_header_(std::move(blob_header)),
      column_names_(std::move(column_names)) {}

absl::StatusOr<Input> Input::CreateFromTensors(std::vector<Tensor> contents,
                                               BlobHeader blob_header) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("No columns provided.");
  }
  if (contents[0].shape().dim_sizes().empty()) {
    return absl::InvalidArgumentError("Column has no rows.");
  }

  size_t num_rows = contents[0].shape().dim_sizes()[0];
  std::vector<std::string> column_names;
  for (const auto& column : contents) {
    column_names.push_back(column.name());
    if (column.shape().dim_sizes().empty()) {
      return absl::InvalidArgumentError("Column has no rows.");
    }
    if (column.shape().dim_sizes().size() > 1) {
      return absl::InvalidArgumentError("Column has more than one dimension.");
    }
    if (column.shape().dim_sizes()[0] != num_rows) {
      return absl::InvalidArgumentError(
          "All columns must have the same number of rows.");
    }
  }
  return Input(TensorContents(std::move(contents)), std::move(blob_header),
               std::move(column_names));
}

absl::Span<const std::string> Input::GetColumnNames() const {
  return column_names_;
}

absl::StatusOr<RowView> Input::GetRow(uint32_t row_index) const {
  return absl::visit(
      [row_index](const auto& data) { return data.GetRow(row_index); },
      contents_);
}

void Input::AddColumn(Tensor&& new_column) {
  column_names_.push_back(new_column.name());
  return absl::visit(
      [new_column = std::move(new_column)](auto& data) mutable {
        data.AddColumn(std::move(new_column));
      },
      contents_);
}

size_t Input::GetRowCount() const {
  return absl::visit(
      [](const auto& data) -> size_t { return data.GetRowCount(); }, contents_);
}

absl::StatusOr<std::vector<Tensor>> Input::MoveToTensors() && {
  return absl::visit(
      [this](auto&& data) -> absl::StatusOr<std::vector<Tensor>> {
        return std::move(data).MoveToTensors(column_names_);
      },
      std::move(contents_));
}

size_t Input::TensorContents::GetRowCount() const {
  if (contents_.empty()) {
    return 0;
  }
  return contents_[0].shape().dim_sizes()[0];
}

absl::StatusOr<Input> Input::CreateFromMessages(
    std::vector<std::unique_ptr<google::protobuf::Message>> messages,
    std::vector<Tensor> system_columns, BlobHeader blob_header) {
  FCP_RETURN_IF_ERROR(ValidateMessageRows(messages, system_columns));
  std::vector<std::string> column_names;
  for (int i = 0; i < messages[0]->GetDescriptor()->field_count(); ++i) {
    column_names.push_back(messages[0]->GetDescriptor()->field(i)->name());
  }
  for (const auto& system_column : system_columns) {
    column_names.push_back(system_column.name());
  }
  return Input(MessageContents(std::move(messages), std::move(system_columns)),
               std::move(blob_header), std::move(column_names));
}

absl::StatusOr<RowView> Input::MessageContents::GetRow(
    uint32_t row_index) const {
  if (row_index >= messages_.size()) {
    return absl::InvalidArgumentError("Row index is out of bounds.");
  }
  return RowView::CreateFromMessage(messages_[row_index].get(), system_columns_,
                                    row_index);
}

absl::StatusOr<std::vector<Tensor>> Input::MessageContents::MoveToTensors(
    absl::Span<const std::string> column_names) && {
  if (messages_.empty()) {
    return std::vector<Tensor>{};
  }

  // The contents of the Message must be copied due to the constraints of the
  // reflection API.
  size_t num_rows = messages_.size();
  size_t num_message_columns = messages_[0]->GetDescriptor()->field_count();
  std::vector<RowView> row_views;
  row_views.reserve(num_rows);
  for (size_t i = 0; i < num_rows; ++i) {
    FCP_ASSIGN_OR_RETURN(
        RowView row_view,
        RowView::CreateFromMessage(messages_[i].get(), system_columns_, i));
    row_views.push_back(row_view);
  }

  std::vector<Tensor> tensors;
  tensors.reserve(row_views[0].GetColumnCount());
  TensorShape shape({static_cast<int64_t>(num_rows)});

  // Create a tensor for each Message-based column by creating a TensorData,
  // populating it, and then creating the tensor.
  for (size_t i = 0; i < num_message_columns; ++i) {
    auto dtype = row_views[0].GetColumnType(i);
    std::unique_ptr<TensorData> tensor_data;
    switch (dtype) {
      case tensorflow_federated::aggregation::DT_INT32:
        tensor_data = CreateVectorTensorData<int32_t>(num_rows, row_views, i);
        break;
      case tensorflow_federated::aggregation::DT_INT64:
        tensor_data = CreateVectorTensorData<int64_t>(num_rows, row_views, i);
        break;
      case tensorflow_federated::aggregation::DT_FLOAT:
        tensor_data = CreateVectorTensorData<float>(num_rows, row_views, i);
        break;
      case tensorflow_federated::aggregation::DT_DOUBLE:
        tensor_data = CreateVectorTensorData<double>(num_rows, row_views, i);
        break;
      case tensorflow_federated::aggregation::DT_STRING: {
        auto builder = std::make_unique<MutableStringData>(num_rows);
        for (const auto& row_view : row_views) {
          builder->Add(std::string(row_view.GetValue<absl::string_view>(i)));
        }
        tensor_data = std::move(builder);
        break;
      }
      default:
        return absl::InvalidArgumentError("Unsupported column type.");
    }
    FCP_ASSIGN_OR_RETURN(
        Tensor tensor,
        Tensor::Create(dtype, shape, std::move(tensor_data), column_names[i]));
    tensors.push_back(std::move(tensor));
  }

  // Move the system columns to the end of the tensor list.
  for (size_t i = 0; i < system_columns_.size(); ++i) {
    tensors.push_back(std::move(system_columns_[i]));
  }

  return tensors;
}

}  // namespace confidential_federated_compute::sql
