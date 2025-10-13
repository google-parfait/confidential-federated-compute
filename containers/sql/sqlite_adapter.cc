// Copyright 2024 Google LLC.
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
#include "containers/sql/sqlite_adapter.h"

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "containers/sql/row_set.h"
#include "containers/sql/row_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "fcp/protos/data_type.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "sqlite3.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"

namespace confidential_federated_compute::sql {

using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::TableSchema;
using ::google::protobuf::RepeatedPtrField;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorData;
namespace {

// A wrapper around `sqlite_stmt*` to automatically call `sqlite3_finalize()`
// when it goes out of scope (go/RAII pattern)
class StatementFinalizer final {
 public:
  explicit StatementFinalizer(sqlite3_stmt* stmt) : stmt_(stmt) {}
  ~StatementFinalizer() { sqlite3_finalize(stmt_); }

  StatementFinalizer(const StatementFinalizer&) = delete;
  StatementFinalizer& operator=(const StatementFinalizer&) = delete;

 private:
  sqlite3_stmt* stmt_;
};

// Binds a value to a prepared SQLite statement.
//
// Uses SQLite's C interface to bind values:
// https://www.sqlite.org/c3ref/bind_blob.html
//
// Takes in the following inputs:
//
// stmt: A prepared SQLite statement (ie. `sqlite3_prepare_v2()` was previously
// called on this statement.)
//
// ordinal: The index of the SQL parameter to be set. The leftmost SQL parameter
// has an index of 1.
//
// row: The row containing the value to bind.
//
// col_index: The index of the column in the row.
//
// status_util: Utility for inspecting SQLite result codes and translating them
// `absl::Status`.
inline absl::Status BindValue(sqlite3_stmt* stmt, int ordinal,
                              const RowView& row, int col_index,
                              const SqliteResultHandler& status_util) {
  int bind_result;
  switch (row.GetColumnType(col_index)) {
    case DataType::DT_INT32:
      bind_result =
          sqlite3_bind_int(stmt, ordinal, row.GetValue<int32_t>(col_index));
      break;
    case DataType::DT_INT64:
      bind_result =
          sqlite3_bind_int64(stmt, ordinal, row.GetValue<int64_t>(col_index));
      break;
    case DataType::DT_FLOAT:
      bind_result =
          sqlite3_bind_double(stmt, ordinal, row.GetValue<float>(col_index));
      break;
    case DataType::DT_DOUBLE:
      bind_result =
          sqlite3_bind_double(stmt, ordinal, row.GetValue<double>(col_index));
      break;
    case DataType::DT_STRING: {
      absl::string_view string_value =
          row.GetValue<absl::string_view>(col_index);
      bind_result = sqlite3_bind_blob(stmt, ordinal, string_value.data(),
                                      string_value.size(), SQLITE_TRANSIENT);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Column type: %d is not a valid column type, can't "
                          "bind value to this column.",
                          row.GetColumnType(col_index)));
  }
  return status_util.ToStatus(bind_result);
}

// Validate that the output columns for the prepared SQL statement match the
// user-specified output schema.
absl::Status ValidateQueryOutputColumns(
    sqlite3_stmt* stmt, const RepeatedPtrField<ColumnSchema>& columns) {
  int output_column_count = sqlite3_column_count(stmt);
  bool is_valid = output_column_count == columns.size();

  std::vector<std::string> output_column_names(output_column_count);
  for (int i = 0; i < output_column_count; ++i) {
    output_column_names[i] = sqlite3_column_name(stmt, i);
    is_valid &=
        columns.size() > i && output_column_names[i] == columns.at(i).name();
  }

  if (!is_valid) {
    std::vector<std::string> schema_field_names;
    for (const auto& schema_col : columns) {
      schema_field_names.push_back(schema_col.name());
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Query results did not match the specified output "
                        "columns.\nSpecified "
                        "output columns: %s\nActual results schema: %s",
                        absl::StrJoin(schema_field_names, ", "),
                        absl::StrJoin(output_column_names, ", ")));
  }
  return absl::OkStatus();
}

template <typename T>
void StoreValue(TensorData* tensor_data, T value) {
  static_cast<MutableVectorData<T>*>(tensor_data)
      ->emplace_back(std::move(value));
}

template <>
void StoreValue(TensorData* tensor_data, std::string value) {
  static_cast<MutableStringData*>(tensor_data)->Add(std::move(value));
}

// Read the result value of a SQLite statement at the specified column index.
//
// The result value is appended to the `TensorData* result` column.
absl::Status ReadSqliteColumn(
    sqlite3_stmt* stmt, google::internal::federated::plan::DataType column_type,
    int column_index, const SqliteResultHandler& status_util,
    TensorData* result) {
  switch (column_type) {
    case google::internal::federated::plan::INT32:
      StoreValue<int32_t>(result, sqlite3_column_int(stmt, column_index));
      break;
    case google::internal::federated::plan::INT64:
      StoreValue<int64_t>(result, sqlite3_column_int64(stmt, column_index));
      break;
    case google::internal::federated::plan::BOOL:
      StoreValue<int32_t>(result, sqlite3_column_int(stmt, column_index) == 1);
      break;
    case google::internal::federated::plan::FLOAT:
      StoreValue<float>(result, sqlite3_column_double(stmt, column_index));
      break;
    case google::internal::federated::plan::DOUBLE:
      StoreValue<double>(result, sqlite3_column_double(stmt, column_index));
      break;
    case google::internal::federated::plan::BYTES: {
      std::string bytes_value(
          static_cast<const char*>(sqlite3_column_blob(stmt, column_index)),
          sqlite3_column_bytes(stmt, column_index));
      StoreValue<std::string>(result, std::move(bytes_value));
      break;
    }
    case google::internal::federated::plan::STRING: {
      std::string string_value(reinterpret_cast<const char*>(
                                   sqlite3_column_text(stmt, column_index)),
                               sqlite3_column_bytes(stmt, column_index));
      StoreValue<std::string>(result, std::move(string_value));
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Not a valid column type, can't read this column");
  }
  return absl::OkStatus();
}

absl::StatusOr<DataType> SqlDataTypeToTensorDtype(
    google::internal::federated::plan::DataType sql_type) {
  switch (sql_type) {
    case google::internal::federated::plan::INT32:
      return DataType::DT_INT32;
    case google::internal::federated::plan::INT64:
      return DataType::DT_INT64;
    case google::internal::federated::plan::BOOL:
      return DataType::DT_INT32;
    case google::internal::federated::plan::FLOAT:
      return DataType::DT_FLOAT;
    case google::internal::federated::plan::DOUBLE:
      return DataType::DT_DOUBLE;
    case google::internal::federated::plan::STRING:
    case google::internal::federated::plan::BYTES:
      return DataType::DT_STRING;
    default:
      return absl::InvalidArgumentError("Unsupported column type.");
  }
}

// Escapes a SQL column name for use in an INSERT statement. This function can
// be used to work around the fact that column names cannot be dynamically bound
// to a prepared statement.
void EscapeSqlColumnName(std::string* out, absl::string_view column) {
  // From https://www.sqlite.org/lang_expr.html#literal_values_constants_, a
  // string constant (which includes column names) can be formed by enclosing a
  // string in single quotes, with single quotes within the string encoded as
  // "''".
  absl::StrAppend(out, "'", absl::StrReplaceAll(column, {{"'", "''"}}), "'");
}

}  // namespace

absl::Status SqliteResultHandler::ToStatus(
    int result_code, absl::StatusCode error_status) const {
  if (result_code == SQLITE_OK || result_code == SQLITE_DONE ||
      result_code == SQLITE_ROW) {
    return absl::OkStatus();
  }

  return absl::Status(error_status,
                      absl::StrFormat("%s (%s)", sqlite3_errmsg(db_),
                                      sqlite3_errstr(result_code)));
}

absl::Status SqliteAdapter::Initialize() {
  int init_result = sqlite3_initialize();
  if (init_result != SQLITE_OK) {
    return absl::InternalError(
        absl::StrFormat("Unable to initialize SQLite library: %s",
                        sqlite3_errstr(init_result)));
  }
  return absl::OkStatus();
}

void SqliteAdapter::ShutDown() { sqlite3_shutdown(); }

absl::StatusOr<std::unique_ptr<SqliteAdapter>> SqliteAdapter::Create() {
  sqlite3* db;
  int open_result = sqlite3_open_v2(
      "in_memory_db", &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_MEMORY, nullptr);
  if (open_result != SQLITE_OK) {
    return absl::InternalError(
        absl::StrFormat("Unable to initialize SQLite library: %s",
                        sqlite3_errstr(open_result)));
  }

  // Enabling extended result codes provides more useful context on errors.
  // https://www.sqlite.org/rescode.html#primary_result_codes_versus_extended_result_codes
  sqlite3_extended_result_codes(db, /*onoff=*/1);

  return absl::WrapUnique(new SqliteAdapter(db));
}

SqliteAdapter::~SqliteAdapter() { sqlite3_close(db_); }

absl::Status SqliteAdapter::DefineTable(TableSchema schema) {
  sqlite3_stmt* stmt;
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(
      sqlite3_prepare_v2(db_, schema.create_table_sql().data(),
                         schema.create_table_sql().size(), &stmt, nullptr)));
  StatementFinalizer finalizer(stmt);

  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_step(stmt)));
  table_schema_.emplace(std::move(schema));
  return absl::OkStatus();
}

absl::Status SqliteAdapter::InsertRows(const RowSet& rows,
                                       absl::string_view insert_stmt) {
  sqlite3_stmt* stmt;
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_prepare_v2(
      db_, insert_stmt.data(), insert_stmt.size(), &stmt, nullptr)));
  StatementFinalizer finalizer(stmt);

  int ordinal = 1;
  for (const auto& row : rows) {
    for (int col_num = 0; col_num < row.GetColumnCount(); ++col_num) {
      FCP_RETURN_IF_ERROR(
          BindValue(stmt, ordinal, row, col_num, result_handler_));
      ordinal++;
    }
  }
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_step(stmt)));
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_reset(stmt)));
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_clear_bindings(stmt)));

  return absl::OkStatus();
}

absl::Status SqliteAdapter::AddTableContents(const RowSet& rows) {
  if (!table_schema_.has_value()) {
    return absl::InvalidArgumentError(
        "`DefineTable` must be called before adding to the table contents.");
  }

  if (rows.size() == 0) {
    return absl::OkStatus();
  }

  // Insert each row into the table, using parameterized query syntax:
  // INSERT INTO t (c1, c2, ...) VALUES (?, ?, ...), (?, ?, ...), ...;
  FCP_ASSIGN_OR_RETURN(std::vector<std::string> column_names,
                       rows.GetColumnNames());
  std::string row_template = absl::StrFormat(
      "(%s)",
      absl::StrJoin(std::vector<std::string>(column_names.size(), "?"), ", "));
  std::string insert_stmt_prefix =
      absl::StrFormat("INSERT INTO %s (%s) VALUES ", table_schema_->name(),
                      absl::StrJoin(column_names, ", ", &EscapeSqlColumnName));

  // Determine how many rows we can insert at once without exceeding the
  // limit on the number of variables in a SQLite statement
  int full_batch_size = kSqliteVariableLimit / column_names.size();

  std::string insert_stmt;
  size_t current_row = 0;
  int batch_size = 0;
  while (current_row < rows.size()) {
    int next_batch_size =
        std::min<int>(full_batch_size, rows.size() - current_row);
    if (next_batch_size != batch_size) {
      batch_size = next_batch_size;
      insert_stmt = absl::StrCat(
          insert_stmt_prefix,
          absl::StrJoin(
              std::vector<absl::string_view>(batch_size, row_template), ", "));
    }
    FCP_RETURN_IF_ERROR(
        InsertRows(rows.subspan(current_row, batch_size), insert_stmt));
    current_row += batch_size;
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<Tensor>> SqliteAdapter::EvaluateQuery(
    absl::string_view query,
    const RepeatedPtrField<ColumnSchema>& output_columns) const {
  std::vector<std::unique_ptr<TensorData>> result_columns(
      output_columns.size());
  for (int i = 0; i < output_columns.size(); i++) {
    std::unique_ptr<TensorData> data;
    switch (output_columns.at(i).type()) {
      case google::internal::federated::plan::INT32:
        data = std::make_unique<MutableVectorData<int32_t>>();
        break;
      case google::internal::federated::plan::INT64:
        data = std::make_unique<MutableVectorData<int64_t>>();
        break;
      case google::internal::federated::plan::BOOL:
        data = std::make_unique<MutableVectorData<int32_t>>();
        break;
      case google::internal::federated::plan::FLOAT:
        data = std::make_unique<MutableVectorData<float>>();
        break;
      case google::internal::federated::plan::DOUBLE:
        data = std::make_unique<MutableVectorData<double>>();
        break;
      case google::internal::federated::plan::STRING:
      case google::internal::federated::plan::BYTES:
        data = std::make_unique<MutableStringData>(0);
        break;
      default:
        return absl::InvalidArgumentError("Unsupported column type.");
    }
    result_columns[i] = std::move(data);
  }

  sqlite3_stmt* stmt;
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(
      sqlite3_prepare_v2(db_, query.data(), query.size(), &stmt, nullptr)));
  StatementFinalizer finalizer(stmt);
  FCP_RETURN_IF_ERROR(ValidateQueryOutputColumns(stmt, output_columns));

  // SQLite uses `sqlite3_step()` to iterate over result rows, and
  // `sqlite_column_*()` functions to extract values for each column.
  int num_rows = 0;
  while (true) {
    int step_result = sqlite3_step(stmt);
    FCP_RETURN_IF_ERROR(result_handler_.ToStatus(step_result));
    if (step_result != SQLITE_ROW) {
      break;
    }
    num_rows++;
    for (int i = 0; i < output_columns.size(); ++i) {
      if (sqlite3_column_type(stmt, i) == SQLITE_NULL) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Encountered NULL value for column `%s`in the query result. "
            "SQLite adapter does not support NULL result values.",
            output_columns.at(i).name()));
      }
      FCP_RETURN_IF_ERROR(ReadSqliteColumn(stmt, output_columns.at(i).type(), i,
                                           result_handler_,
                                           result_columns.at(i).get()));
    }
  }
  std::vector<Tensor> result(output_columns.size());

  for (int i = 0; i < output_columns.size(); ++i) {
    FCP_ASSIGN_OR_RETURN(DataType dtype,
                         SqlDataTypeToTensorDtype(output_columns.at(i).type()));
    FCP_ASSIGN_OR_RETURN(
        Tensor column_tensor,
        Tensor::Create(dtype, {num_rows}, std::move(result_columns.at(i)),
                       output_columns.at(i).name()));
    result[i] = std::move(column_tensor);
  }
  return result;
}

}  // namespace confidential_federated_compute::sql
