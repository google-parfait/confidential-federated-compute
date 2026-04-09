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

#include "containers/sql_data_ingress/sql_utils.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "containers/sql/row_set.h"
#include "containers/sql/sqlite_adapter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sql_data_ingress {

using ::confidential_federated_compute::sql::RowLocation;
using ::confidential_federated_compute::sql::RowSet;
using ::confidential_federated_compute::sql::SqliteAdapter;
using ::fcp::confidentialcompute::TableSchema;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::Tensor;

absl::StatusOr<std::vector<Tensor>> Deserialize(const TableSchema& table_schema,
                                                CheckpointParser* checkpoint) {
  std::vector<Tensor> columns;
  columns.reserve(table_schema.column_size());
  std::optional<size_t> num_rows;

  for (int i = 0; i < table_schema.column_size(); i++) {
    auto tensor_or = checkpoint->GetTensor(table_schema.column(i).name());
    if (!tensor_or.ok()) return tensor_or.status();
    Tensor column = std::move(tensor_or).value();

    if (!num_rows.has_value()) {
      num_rows.emplace(column.num_elements());
    } else if (num_rows.value() != column.num_elements()) {
      return absl::InvalidArgumentError(
          "Checkpoint has columns with differing numbers of rows.");
    }
    columns.push_back(std::move(column));
  }
  return columns;
}

absl::StatusOr<std::vector<Tensor>> ExecuteClientQuery(
    const SqlConfiguration& configuration, RowSet rows) {
  auto sqlite_or = SqliteAdapter::Create();
  if (!sqlite_or.ok()) return sqlite_or.status();
  std::unique_ptr<SqliteAdapter> sqlite = std::move(sqlite_or).value();

  auto status = sqlite->DefineTable(configuration.input_schema);
  if (!status.ok()) return status;

  status = sqlite->AddTableContents(rows);
  if (!status.ok()) return status;

  return sqlite->EvaluateQuery(configuration.query,
                               configuration.output_columns);
}

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

}  // namespace confidential_federated_compute::sql_data_ingress
