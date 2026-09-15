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

#include "containers/common/io/tabular/input.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "containers/common/io/checkpoint_utils.h"
#include "containers/common/io/tabular/row_view.h"
#include "fcp/confidentialcompute/constants.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::DynamicMessageFactory;
using ::google::protobuf::FileDescriptorSet;
using ::google::protobuf::Message;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::Tensor;

// Suffix appended to enum column names to produce the string-valued column.
// Used only for naming; the type itself is encoded in
// ColumnDescriptor::column_type.
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
std::vector<T> CreateColumnValues(absl::Span<const RowView> row_views,
                                  size_t column_index) {
  std::vector<T> values;
  values.reserve(row_views.size());
  for (const auto& row_view : row_views) {
    values.push_back(row_view.GetValue<T>(column_index));
  }
  return values;
}

// Maps a proto field's cpp_type to the DataType used for its SQLite column.
// Enum fields map to DT_INT32 (their integer representation).
// Called once at schema-build time; the result is stored in ColumnDescriptor.
DataType GetSQLiteColumnType(const google::protobuf::FieldDescriptor* field) {
  switch (field->cpp_type()) {
    case google::protobuf::FieldDescriptor::CPPTYPE_BOOL:
    case google::protobuf::FieldDescriptor::CPPTYPE_ENUM:
    case google::protobuf::FieldDescriptor::CPPTYPE_INT32:
    case google::protobuf::FieldDescriptor::CPPTYPE_UINT32:
      return DataType::DT_INT32;
    case google::protobuf::FieldDescriptor::CPPTYPE_INT64:
    case google::protobuf::FieldDescriptor::CPPTYPE_UINT64:
      return DataType::DT_INT64;
    case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      return DataType::DT_FLOAT;
    case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      return DataType::DT_DOUBLE;
    case google::protobuf::FieldDescriptor::CPPTYPE_STRING:
      return DataType::DT_STRING;
    default:
      LOG(FATAL) << "Unsupported field type " << field->cpp_type_name();
  }
}

// Recursively flattens a nested protobuf schema into ColumnDescriptor entries.
//
// For each scalar field, appends one ColumnDescriptor to `schema` whose name
// is derived from the traversal path.  For enum fields, appends TWO entries
// with the same proto_path but different names and column_types:
//   [0] column_type = DT_INT32  → the integer value  (e.g. "event_type")
//   [1] column_type = DT_STRING → the enum name       (e.g.
//   "event_type_as_str")
//
// All entries produced here are proto columns (non-empty proto_path).
// System tensor columns are appended by CreateFromMessages after this call.
void GetFlattenedSchema(const Descriptor* descriptor, std::string prefix,
                        FieldPath& current_path, MessageColumnSchema& schema) {
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const google::protobuf::FieldDescriptor* field = descriptor->field(i);
    if (field->is_repeated()) {
      LOG(WARNING) << "Repeated fields are not supported and will be skipped: "
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
                         current_path, schema);
    } else {
      // SQLite column — name and type determined once here.
      schema.push_back({absl::StrCat(prefix, field->name()),
                        GetSQLiteColumnType(field), current_path});

      // For enum fields, we create a second column that maps to the same
      // descriptor to store stringified enum names. This allows us to
      // perform aggregation on the enum field by either its integer or
      // string representation, whichever is more convenient for the user.
      if (field->cpp_type() ==
          google::protobuf::FieldDescriptor::CPPTYPE_ENUM) {
        // String-representation column: same path, explicit DT_STRING type,
        // distinct name with the "_as_str" suffix.
        schema.push_back(
            {absl::StrCat(prefix, field->name(), kEnumAsStringSuffix),
             DataType::DT_STRING, current_path});
      }
    }
    current_path.pop_back();
  }
}

}  // namespace

Input::TensorContents::TensorContents(std::vector<Tensor> contents)
    : contents_(std::move(contents)) {
  column_names_.reserve(contents_.size());
  for (const auto& col : contents_) {
    column_names_.push_back(col.name());
  }
}

Input::Input(ContentsVariant contents, std::string metadata,
             std::optional<std::string> privacy_id)
    : contents_(std::move(contents)),
      metadata_(std::move(metadata)),
      privacy_id_(std::move(privacy_id)) {}

absl::StatusOr<Input> Input::CreateFromTensors(
    std::vector<Tensor> contents, std::string metadata,
    std::optional<std::string> privacy_id) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("No columns provided.");
  }
  if (contents[0].shape().dim_sizes().empty()) {
    return absl::InvalidArgumentError("Column has no rows.");
  }

  size_t num_rows = contents[0].shape().dim_sizes()[0];
  for (const auto& column : contents) {
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
               std::move(privacy_id));
}

absl::StatusOr<RowView> Input::GetRow(uint32_t row_index) const {
  return absl::visit(
      [row_index](const auto& data) { return data.GetRow(row_index); },
      contents_);
}

absl::Status Input::AddColumn(Tensor&& new_column) {
  ABSL_RETURN_IF_ERROR(
      ValidateNewColumn(new_column, GetColumnNames(), GetRowCount()));
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
      [](auto&& data) -> absl::StatusOr<std::vector<Tensor>> {
        return std::move(data).MoveToTensors();
      },
      std::move(contents_));
}

size_t Input::TensorContents::GetRowCount() const {
  if (contents_.empty()) {
    return 0;
  }
  return contents_[0].shape().dim_sizes()[0];
}

Input::MessageContents::MessageContents(
    std::vector<std::unique_ptr<Message>> messages,
    std::vector<Tensor> system_columns, MessageColumnSchema schema)
    : messages_(std::move(messages)),
      system_columns_(std::move(system_columns)),
      schema_(std::move(schema)) {
  column_names_.reserve(schema_.size());
  for (const auto& desc : schema_) {
    column_names_.push_back(desc.name);
  }
}

void Input::MessageContents::AddColumn(Tensor&& column) {
  size_t tensor_index = system_columns_.size();
  schema_.push_back(
      {column.name(), column.dtype(), /*proto_path=*/{}, tensor_index});
  column_names_.push_back(column.name());
  system_columns_.push_back(std::move(column));
}

absl::StatusOr<Input> Input::CreateFromMessages(
    std::vector<std::unique_ptr<Message>> messages,
    std::vector<Tensor> system_columns, std::string metadata,
    std::optional<std::string> privacy_id) {
  ABSL_RETURN_IF_ERROR(ValidateMessageRows(messages, system_columns));

  // Build schema for proto columns.
  FieldPath current_path;
  MessageColumnSchema schema;
  GetFlattenedSchema(messages[0]->GetDescriptor(), "", current_path, schema);

  // Append system tensor columns to the schema. Each gets a ColumnDescriptor
  // with an empty proto_path and a system_tensor_index pointing into the
  // system_columns vector inside MessageContents.
  for (size_t i = 0; i < system_columns.size(); ++i) {
    schema.push_back({system_columns[i].name(), system_columns[i].dtype(),
                      /*proto_path=*/{},
                      /*system_tensor_index=*/i});
  }

  return Input(MessageContents(std::move(messages), std::move(system_columns),
                               std::move(schema)),
               std::move(metadata), std::move(privacy_id));
}

absl::StatusOr<RowView> Input::MessageContents::GetRow(
    uint32_t row_index) const {
  if (row_index >= messages_.size()) {
    return absl::InvalidArgumentError("Row index is out of bounds.");
  }
  return RowView::CreateFromMessage(messages_[row_index].get(), system_columns_,
                                    row_index, &schema_);
}

absl::StatusOr<std::vector<Tensor>> Input::MessageContents::MoveToTensors() && {
  if (messages_.empty()) {
    return std::vector<Tensor>{};
  }

  // The contents of the Message must be copied due to the constraints of the
  // reflection API.
  size_t num_rows = messages_.size();
  std::vector<RowView> row_views;
  row_views.reserve(num_rows);
  for (size_t i = 0; i < num_rows; ++i) {
    ABSL_ASSIGN_OR_RETURN(RowView row_view, RowView::CreateFromMessage(
                                                messages_[i].get(),
                                                system_columns_, i, &schema_));
    row_views.push_back(row_view);
  }

  std::vector<Tensor> tensors;
  tensors.reserve(row_views[0].GetColumnCount());

  // Iterate the unified schema: proto columns first, then system tensor
  // columns. GetColumnType(i) reads ColumnDescriptor::column_type directly —
  // no special-casing for enum-as-string or system columns needed here.
  for (size_t i = 0; i < schema_.size(); ++i) {
    const ColumnDescriptor& desc = schema_[i];

    if (!desc.proto_path.empty()) {
      // Proto column: serialize values row-by-row via RowView using the
      // Tensor 1D constructor.
      auto dtype = row_views[0].GetColumnType(i);
      Tensor tensor;
      switch (dtype) {
        case tensorflow_federated::aggregation::DT_INT32:
          tensor = Tensor(CreateColumnValues<int32_t>(row_views, i), desc.name);
          break;
        case tensorflow_federated::aggregation::DT_INT64:
          tensor = Tensor(CreateColumnValues<int64_t>(row_views, i), desc.name);
          break;
        case tensorflow_federated::aggregation::DT_FLOAT:
          tensor = Tensor(CreateColumnValues<float>(row_views, i), desc.name);
          break;
        case tensorflow_federated::aggregation::DT_DOUBLE:
          tensor = Tensor(CreateColumnValues<double>(row_views, i), desc.name);
          break;
        case tensorflow_federated::aggregation::DT_STRING: {
          std::vector<std::string> values;
          values.reserve(row_views.size());
          for (const auto& row_view : row_views) {
            values.push_back(
                std::string(row_view.GetValue<absl::string_view>(i)));
          }
          tensor = Tensor(std::move(values), desc.name);
          break;
        }
        default:
          return absl::InvalidArgumentError("Unsupported column type.");
      }
      tensors.push_back(std::move(tensor));
    } else {
      // System tensor column: move the tensor directly.
      tensors.push_back(std::move(system_columns_[desc.system_tensor_index]));
    }
  }

  return tensors;
}

absl::StatusOr<Input> CreateFromMessageCheckpoint(
    tensorflow_federated::aggregation::CheckpointParser* checkpoint,
    MessageFactory& message_factory, absl::string_view on_device_query_name) {
  std::string column_prefix = absl::StrCat(on_device_query_name, "/");
  ABSL_ASSIGN_OR_RETURN(
      Tensor entry_tensor,
      checkpoint->GetTensor(absl::StrCat(
          column_prefix, fcp::confidential_compute::kPrivateLoggerEntryKey)));
  if (entry_tensor.dtype() !=
      tensorflow_federated::aggregation::DataType::DT_STRING) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`%s` tensor must be a string tensor",
                        fcp::confidential_compute::kPrivateLoggerEntryKey));
  }
  ABSL_ASSIGN_OR_RETURN(Tensor time_tensor,
                        GetEventTime(*checkpoint, on_device_query_name));

  // Rename the time tensor to remove the column prefix. Pipelines that process
  // Message-based checkpoints don't use the column name prefix.
  ABSL_RETURN_IF_ERROR(
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

  // Extract privacy ID if present
  std::optional<std::string> privacy_id;
  absl::StatusOr<std::string> pid_result = GetPrivacyId(*checkpoint);
  if (pid_result.ok()) {
    privacy_id = *std::move(pid_result);
  } else if (!absl::IsNotFound(pid_result.status())) {
    // The tensor exists but is invalid.
    return pid_result.status();
  }

  std::vector<Tensor> system_columns;
  system_columns.reserve(1);
  system_columns.push_back(std::move(time_tensor));
  return Input::CreateFromMessages(std::move(messages),
                                   std::move(system_columns),
                                   /*metadata=*/"", std::move(privacy_id));
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
