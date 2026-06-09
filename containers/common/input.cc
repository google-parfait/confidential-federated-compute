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

#include "containers/common/input.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "containers/common/row_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/constants.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::DynamicMessageFactory;
using ::google::protobuf::FileDescriptorSet;
using ::google::protobuf::Message;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorData;
using ::tensorflow_federated::aggregation::TensorShape;

// Suffix appended to enum fields in schema to hold string representation
constexpr char kEnumAsStringSuffix[] = "_as_str";

absl::Status ValidateMessageRows(
    absl::Span<const std::unique_ptr<Message>> messages,
    absl::Span<const Tensor> system_columns) {
  if (messages.empty()) {
    return absl::InvalidArgumentError("No rows provided.");
  }
  const auto& first_row = messages[0];
  const Descriptor* first_descriptor = first_row->GetDescriptor();
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

absl::Status ValidateNewColumn(const Tensor& new_column,
                               absl::Span<const std::string> column_names,
                               size_t row_count) {
  if (new_column.name().empty()) {
    return absl::InvalidArgumentError("Column name is empty.");
  }
  if (std::find(column_names.begin(), column_names.end(), new_column.name()) !=
      column_names.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column name ", new_column.name(), " already exists."));
  }
  if (new_column.shape().dim_sizes().size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Column ", new_column.name(), " must have exactly one dimension."));
  }
  if (new_column.shape().dim_sizes()[0] != row_count) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column ", new_column.name(),
                     " has a different number of rows than the table."));
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

// Recursively flattens a nested protobuf schema into flat table columns.
//
// For each scalar field, produces a column name based on the traversed path
// and pushes the sequence of FieldDescriptor pointers that can locate the
// field value within messages into `field_paths`. This computation happens
// once during table initialization, avoiding repeating path calculations per
// row.
void GetFlattenedSchema(const Descriptor* descriptor, std::string prefix,
                        FieldPath& current_path,
                        std::vector<std::string>& column_names,
                        FieldPathList& field_paths) {
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const google::protobuf::FieldDescriptor* field = descriptor->field(i);
    if (field->is_repeated()) {
      FCP_LOG(WARNING)
          << "Repeated fields are not supported and will be skipped: "
          << field->full_name();
      continue;
    }
    current_path.push_back(field);
    if (field->cpp_type() ==
        google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
      // Use double underscore separator to flatten nested fields to avoid SQL
      // identifier quoting issues in SQLite.
      GetFlattenedSchema(field->message_type(),
                         absl::StrCat(prefix, field->name(), "__"),
                         current_path, column_names, field_paths);
    } else {
      column_names.push_back(absl::StrCat(prefix, field->name()));
      field_paths.push_back(current_path);
      // For enum fields, we create a second column that maps to the same
      // descriptor to store string representation. This allows consumers to
      // read enum descriptions instead of codes.
      if (field->cpp_type() ==
          google::protobuf::FieldDescriptor::CPPTYPE_ENUM) {
        column_names.push_back(
            absl::StrCat(prefix, field->name(), kEnumAsStringSuffix));
        field_paths.push_back(current_path);
      }
    }
    current_path.pop_back();
  }
}

}  // namespace
Input::Input(ContentsVariant contents, std::string metadata,
             std::vector<std::string> column_names,
             std::optional<std::string> privacy_id)
    : contents_(std::move(contents)),
      metadata_(std::move(metadata)),
      column_names_(std::move(column_names)),
      privacy_id_(std::move(privacy_id)) {}

absl::StatusOr<std::optional<std::string>> ExtractPrivacyIdAndValidate(
    const std::optional<Tensor>& privacy_id) {
  if (!privacy_id.has_value()) {
    return std::nullopt;
  }
  if (privacy_id->dtype() != tensorflow_federated::aggregation::DT_STRING) {
    return absl::InvalidArgumentError("Privacy ID must be of type DT_STRING.");
  }
  FCP_ASSIGN_OR_RETURN(int64_t num_elements, privacy_id->shape().NumElements());
  if (num_elements != 1) {
    return absl::InvalidArgumentError("Privacy ID must be a scalar.");
  }
  return std::string(privacy_id->AsScalar<absl::string_view>());
}

absl::StatusOr<Input> Input::CreateFromTensors(
    std::vector<Tensor> contents, std::string metadata,
    std::optional<Tensor> privacy_id) {
  FCP_ASSIGN_OR_RETURN(std::optional<std::string> privacy_id_string,
                       ExtractPrivacyIdAndValidate(privacy_id));
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
  return Input(TensorContents(std::move(contents)), std::move(metadata),
               std::move(column_names), std::move(privacy_id_string));
}

absl::Span<const std::string> Input::GetColumnNames() const {
  return column_names_;
}

absl::StatusOr<RowView> Input::GetRow(uint32_t row_index) const {
  return absl::visit(
      [row_index](const auto& data) { return data.GetRow(row_index); },
      contents_);
}

absl::Status Input::AddColumn(Tensor&& new_column) {
  FCP_RETURN_IF_ERROR(
      ValidateNewColumn(new_column, column_names_, GetRowCount()));
  column_names_.push_back(new_column.name());
  absl::visit(
      [new_column = std::move(new_column)](auto& data) mutable {
        data.AddColumn(std::move(new_column));
      },
      contents_);
  return absl::OkStatus();
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
    std::vector<std::unique_ptr<Message>> messages,
    std::vector<Tensor> system_columns, std::string metadata,
    std::optional<Tensor> privacy_id) {
  FCP_ASSIGN_OR_RETURN(std::optional<std::string> privacy_id_string,
                       ExtractPrivacyIdAndValidate(privacy_id));
  FCP_RETURN_IF_ERROR(ValidateMessageRows(messages, system_columns));
  std::vector<std::string> column_names;
  FieldPath current_path;
  FieldPathList field_paths;
  GetFlattenedSchema(messages[0]->GetDescriptor(), "", current_path,
                     column_names, field_paths);
  for (const auto& system_column : system_columns) {
    column_names.push_back(system_column.name());
  }
  return Input(MessageContents(std::move(messages), std::move(system_columns),
                               std::move(field_paths)),
               std::move(metadata), std::move(column_names),
               std::move(privacy_id_string));
}

absl::StatusOr<RowView> Input::MessageContents::GetRow(
    uint32_t row_index) const {
  if (row_index >= messages_.size()) {
    return absl::InvalidArgumentError("Row index is out of bounds.");
  }
  return RowView::CreateFromMessage(messages_[row_index].get(), system_columns_,
                                    row_index, &field_paths_);
}

absl::StatusOr<std::vector<Tensor>> Input::MessageContents::MoveToTensors(
    absl::Span<const std::string> column_names) && {
  if (messages_.empty()) {
    return std::vector<Tensor>{};
  }

  // The contents of the Message must be copied due to the constraints of the
  // reflection API.
  size_t num_rows = messages_.size();
  std::vector<RowView> row_views;
  row_views.reserve(num_rows);
  for (size_t i = 0; i < num_rows; ++i) {
    FCP_ASSIGN_OR_RETURN(
        RowView row_view,
        RowView::CreateFromMessage(messages_[i].get(), system_columns_, i,
                                   &field_paths_));
    row_views.push_back(row_view);
  }
  size_t num_message_columns =
      row_views[0].GetColumnCount() - system_columns_.size();

  std::vector<Tensor> tensors;
  tensors.reserve(row_views[0].GetColumnCount());
  TensorShape shape({static_cast<int64_t>(num_rows)});

  // Create a tensor for each Message-based column by creating a TensorData,
  // populating it, and then creating the tensor.
  for (size_t i = 0; i < num_message_columns; ++i) {
    auto dtype = row_views[0].GetColumnType(i);
    const google::protobuf::FieldDescriptor* field = field_paths_[i].back();
    // We add a string mapping here so that when we parse the string and the
    // type is enum we use the enum name() instead of mapped integer codes.
    if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_ENUM &&
        absl::EndsWith(column_names[i], kEnumAsStringSuffix)) {
      dtype = tensorflow_federated::aggregation::DataType::DT_STRING;
    }
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

absl::StatusOr<Input> CreateFromMessageCheckpoint(
    std::string metadata,
    tensorflow_federated::aggregation::CheckpointParser* checkpoint,
    MessageFactory& message_factory, absl::string_view on_device_query_name) {
  std::string column_prefix = absl::StrCat(on_device_query_name, "/");
  FCP_ASSIGN_OR_RETURN(
      Tensor entry_tensor,
      checkpoint->GetTensor(absl::StrCat(
          column_prefix, fcp::confidential_compute::kPrivateLoggerEntryKey)));
  if (entry_tensor.dtype() !=
      tensorflow_federated::aggregation::DataType::DT_STRING) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`%s` tensor must be a string tensor",
                        fcp::confidential_compute::kPrivateLoggerEntryKey));
  }
  FCP_ASSIGN_OR_RETURN(
      Tensor time_tensor,
      checkpoint->GetTensor(absl::StrCat(
          column_prefix, fcp::confidential_compute::kEventTimeColumnName)));
  if (time_tensor.dtype() !=
      tensorflow_federated::aggregation::DataType::DT_STRING) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`%s` tensor must be a string tensor",
                        fcp::confidential_compute::kEventTimeColumnName));
  }

  // Rename the time tensor to remove the column prefix. Pipelines that process
  // Message-based checkpoints don't use the column name prefix.
  FCP_RETURN_IF_ERROR(
      time_tensor.set_name(fcp::confidential_compute::kEventTimeColumnName));

  std::vector<std::unique_ptr<Message>> messages;
  messages.reserve(entry_tensor.num_elements());
  for (const absl::string_view entry :
       entry_tensor.AsSpan<absl::string_view>()) {
    std::unique_ptr<Message> message(message_factory.NewMessage());
    if (!message->ParseFromString(entry)) {
      // Note that ParseFrom* methods are documented as calling Clear() on the
      // message before parsing. Thus it's fine if the failed ParseFromString
      // above leaves the message in a partial state.
      if (!message->ParseFromArray(entry.data(), entry.size())) {
        return absl::InvalidArgumentError("Failed to parse proto");
      }
    }
    messages.push_back(std::move(message));
  }

  std::vector<Tensor> system_columns;
  system_columns.reserve(1);
  system_columns.push_back(std::move(time_tensor));
  return Input::CreateFromMessages(
      std::move(messages), std::move(system_columns), std::move(metadata));
}

absl::StatusOr<std::unique_ptr<MessageFactory>>
FileDescriptorSetMessageFactory::Create(
    const FileDescriptorSet& file_descriptor_set,
    absl::string_view message_name) {
  std::unique_ptr<DescriptorPool> descriptor_pool =
      std::make_unique<DescriptorPool>();
  for (const auto& file_descriptor_proto : file_descriptor_set.file()) {
    if (descriptor_pool->BuildFile(file_descriptor_proto) == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to build file descriptor for ",
                       file_descriptor_proto.name()));
    }
  }

  const Descriptor* message_descriptor =
      descriptor_pool->FindMessageTypeByName(message_name);
  if (message_descriptor == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not find message '", message_name,
                     "' in the provided descriptor set."));
  }
  std::unique_ptr<DynamicMessageFactory> dynamic_message_factory =
      std::make_unique<DynamicMessageFactory>(descriptor_pool.get());
  const Message* prototype =
      dynamic_message_factory->GetPrototype(message_descriptor);
  if (prototype == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not create prototype for message '", message_name,
                     "' from the provided descriptor set."));
  }
  return absl::WrapUnique(new FileDescriptorSetMessageFactory(
      std::move(descriptor_pool), std::move(dynamic_message_factory),
      prototype));
}

}  // namespace confidential_federated_compute
