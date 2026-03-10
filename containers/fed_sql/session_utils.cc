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

#include "containers/fed_sql/session_utils.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "containers/crypto.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::confidential_federated_compute::sql::Input;
using ::confidential_federated_compute::sql::RowLocation;
using ::confidential_federated_compute::sql::RowSet;
using ::confidential_federated_compute::sql::SqliteAdapter;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivateLoggerEntryKey;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::InferenceConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::TableSchema;
using ::google::protobuf::Message;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

}  // namespace

absl::StatusOr<std::vector<Tensor>> Deserialize(
    const TableSchema& table_schema, CheckpointParser* checkpoint,
    std::optional<InferenceConfiguration> inference_configuration) {
  absl::flat_hash_set<std::string> input_column_names;
  absl::flat_hash_set<std::string> output_column_names;
  if (inference_configuration.has_value()) {
    for (const auto& inference_task :
         inference_configuration->inference_task()) {
      const auto& input_names =
          inference_task.column_config().input_column_names();
      input_column_names.insert(input_names.begin(), input_names.end());
      output_column_names.insert(
          inference_task.column_config().output_column_name());
    }
  }

  // The table schema has inference output columns listed, but the checkpoint
  // does not contain these, as they are generated during inference. Besides
  // that, input columns are not listed in the table schema, but are present in
  // the checkpoint, and have to be unpacked.
  std::vector<Tensor> columns(table_schema.column_size() -
                              output_column_names.size() +
                              input_column_names.size());
  std::optional<size_t> num_rows;
  int tensor_column_index = 0;

  for (int i = 0; i < table_schema.column_size(); i++) {
    if (output_column_names.contains(table_schema.column(i).name())) {
      // Inference output columns do not exist in the checkpoint.
      continue;
    }
    TFF_ASSIGN_OR_RETURN(Tensor column,
                         checkpoint->GetTensor(table_schema.column(i).name()));
    if (!num_rows.has_value()) {
      num_rows.emplace(column.num_elements());
    } else if (num_rows.value() != column.num_elements()) {
      return absl::InvalidArgumentError(
          "Checkpoint has columns with differing numbers of rows.");
    }
    columns[tensor_column_index] = std::move(column);
    tensor_column_index++;
  }
  for (const auto& column_name : input_column_names) {
    // Input columns are not listed in the per-client table schema, and have to
    // be added manually.
    TFF_ASSIGN_OR_RETURN(Tensor column, checkpoint->GetTensor(column_name));
    if (!num_rows.has_value()) {
      num_rows.emplace(column.num_elements());
    } else if (num_rows.value() != column.num_elements()) {
      return absl::InvalidArgumentError(
          "Checkpoint has columns with differing numbers of rows.");
    }
    columns[tensor_column_index] = std::move(column);
    tensor_column_index++;
  }
  return columns;
}

// Creates a RowLocation for each row in an input that contains `num_rows`.
std::vector<RowLocation> CreateRowLocationsForAllRows(size_t num_rows) {
  if (num_rows == 0) {
    return {};
  }
  std::vector<RowLocation> locations;
  locations.reserve(num_rows);
  for (uint32_t i = 0; i < num_rows; ++i) {
    locations.push_back({.dp_unit_hash = 0, .input_index = 0, .row_index = i});
  }
  return locations;
}

absl::StatusOr<std::vector<Tensor>> ExecuteClientQuery(
    const SqlConfiguration& configuration, RowSet rows) {
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<SqliteAdapter> sqlite,
                       SqliteAdapter::Create());
  FCP_RETURN_IF_ERROR(sqlite->DefineTable(configuration.input_schema));
  FCP_RETURN_IF_ERROR(sqlite->AddTableContents(rows));
  return sqlite->EvaluateQuery(configuration.query,
                               configuration.output_columns);
}

absl::StatusOr<Input> CreateInputFromMessageCheckpoint(
    BlobHeader blob_header, CheckpointParser* checkpoint,
    MessageFactory& message_factory, absl::string_view on_device_query_name) {
  std::string column_prefix = absl::StrCat(on_device_query_name, "/");
  FCP_ASSIGN_OR_RETURN(Tensor entry_tensor,
                       checkpoint->GetTensor(absl::StrCat(
                           column_prefix, kPrivateLoggerEntryKey)));
  if (entry_tensor.dtype() !=
      tensorflow_federated::aggregation::DataType::DT_STRING) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must be a string tensor", kPrivateLoggerEntryKey));
  }
  FCP_ASSIGN_OR_RETURN(
      Tensor time_tensor,
      checkpoint->GetTensor(absl::StrCat(column_prefix, kEventTimeColumnName)));
  if (time_tensor.dtype() !=
      tensorflow_federated::aggregation::DataType::DT_STRING) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must be a string tensor", kEventTimeColumnName));
  }

  // Rename the time tensor to remove the column prefix. Pipelines that process
  // Message-based checkpoints don't use the column name prefix.
  FCP_RETURN_IF_ERROR(time_tensor.set_name(kEventTimeColumnName));

  std::vector<std::unique_ptr<google::protobuf::Message>> messages;
  messages.reserve(entry_tensor.num_elements());
  for (const absl::string_view entry :
       entry_tensor.AsSpan<absl::string_view>()) {
    std::unique_ptr<google::protobuf::Message> message(
        message_factory.NewMessage());
    // Try to base64 decode the entry. Old versions of the client base64
    // encode entries. We try to base64 decode first since it's extremely
    // unlikely for a valid binary proto to be a valid Base64 string (while the
    // inverse is more likely).
    std::string decoded_entry;
    if (!absl::Base64Unescape(entry, &decoded_entry) ||
        !message->ParseFromString(decoded_entry)) {
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
  return Input::CreateFromMessages(std::move(messages),
                                   std::move(system_columns), blob_header);
}

}  // namespace confidential_federated_compute::fed_sql
