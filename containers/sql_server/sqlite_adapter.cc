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
#include "containers/sql_server/sqlite_adapter.h"

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "containers/sql_server/sql_data.pb.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/example_query_result.pb.h"
#include "sqlite3.h"

namespace confidential_federated_compute::sql_server {

using ::fcp::client::ExampleQueryResult_VectorData;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::sql_data::ColumnSchema;
using ::sql_data::SqlData;
using ::sql_data::SqlQuery;
using ::sql_data::TableSchema;

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
// row_num: The index of the value to bind in `column_values`.
//
// column_values: A `Values` representing a column.
//
// column_type: The data type of `column_values`.
//
// status_util: Utility for inspecting SQLite result codes and translating them
// `absl::Status`.
absl::Status BindSqliteParameter(
    sqlite3_stmt* stmt, int ordinal, int row_num,
    ExampleQueryResult_VectorData_Values column_values,
    ColumnSchema::DataType column_type,
    const SqliteResultHandler& status_util) {
  switch (column_type) {
    case ColumnSchema::INT32:
      return status_util.ToStatus(sqlite3_bind_int(
          stmt, ordinal, column_values.int32_values().value(row_num)));
    case ColumnSchema::INT64:
      return status_util.ToStatus(sqlite3_bind_int64(
          stmt, ordinal, column_values.int64_values().value(row_num)));
    case ColumnSchema::BOOL:
      return status_util.ToStatus(sqlite3_bind_int(
          stmt, ordinal, column_values.bool_values().value(row_num)));
    case ColumnSchema::FLOAT:
      return status_util.ToStatus(sqlite3_bind_double(
          stmt, ordinal, column_values.float_values().value(row_num)));
    case ColumnSchema::DOUBLE:
      return status_util.ToStatus(sqlite3_bind_double(
          stmt, ordinal, column_values.double_values().value(row_num)));
    case ColumnSchema::BYTES: {
      std::string bytes_value = column_values.bytes_values().value(row_num);
      return status_util.ToStatus(
          sqlite3_bind_blob(stmt, ordinal, bytes_value.data(),
                            bytes_value.size(), SQLITE_TRANSIENT));
    }
    case ColumnSchema::STRING: {
      std::string string_value = column_values.string_values().value(row_num);
      return status_util.ToStatus(
          sqlite3_bind_text(stmt, ordinal, string_value.data(),
                            string_value.size(), SQLITE_TRANSIENT));
    }
    default:
      return absl::InvalidArgumentError(
          "Not a valid column type, can't bind value to this column.");
  }
}

// Validate that the output columns for the prepared SQL statement match the
// user-specified output schema.
absl::Status ValidateQueryOutputSchema(sqlite3_stmt* stmt,
                                       const TableSchema& schema) {
  int output_column_count = sqlite3_column_count(stmt);
  bool is_valid = output_column_count == schema.column_size();

  std::vector<std::string> schema_field_names;
  for (const auto& schema_col : schema.column()) {
    schema_field_names.push_back(schema_col.name());
  }

  std::vector<std::string> output_column_names(output_column_count);
  for (int i = 0; i < output_column_count; ++i) {
    output_column_names[i] = sqlite3_column_name(stmt, i);
    is_valid &= schema_field_names.size() > i &&
                output_column_names[i] == schema_field_names[i];
  }

  if (!is_valid) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Query results did not match the specified output schema.\nSpecified "
        "output schema: %s\nActual results schema: %s",
        absl::StrJoin(schema_field_names, ", "),
        absl::StrJoin(output_column_names, ", ")));
  }
  return absl::OkStatus();
}

// Read the result value of a SQLite statement at the specified column index.
//
// The result value is appended to the `ExampleQueryResult_VectorData_Values*
// result` column.
absl::Status ReadSqliteColumn(sqlite3_stmt* stmt,
                              ColumnSchema::DataType column_type,
                              int column_index,
                              const SqliteResultHandler& status_util,
                              ExampleQueryResult_VectorData_Values* result) {
  // TODO: Switch to using an interface that abstracts away the
  // data format.
  switch (column_type) {
    case ColumnSchema::INT32:
      result->mutable_int32_values()->add_value(
          sqlite3_column_int(stmt, column_index));
      break;
    case ColumnSchema::INT64:
      result->mutable_int64_values()->add_value(
          sqlite3_column_int64(stmt, column_index));
      break;
    case ColumnSchema::BOOL:
      result->mutable_bool_values()->add_value(
          sqlite3_column_int(stmt, column_index) == 1);
      break;
    case ColumnSchema::FLOAT:
      result->mutable_float_values()->add_value(
          sqlite3_column_double(stmt, column_index));
      break;
    case ColumnSchema::DOUBLE:
      result->mutable_double_values()->add_value(
          sqlite3_column_double(stmt, column_index));
      break;
    case ColumnSchema::BYTES: {
      std::string bytes_value(
          static_cast<const char*>(sqlite3_column_blob(stmt, column_index)),
          sqlite3_column_bytes(stmt, column_index));
      result->mutable_bytes_values()->add_value(bytes_value);
      break;
    }
    case ColumnSchema::STRING: {
      std::string string_value(reinterpret_cast<const char*>(
                                   sqlite3_column_text(stmt, column_index)),
                               sqlite3_column_bytes(stmt, column_index));
      result->mutable_string_values()->add_value(string_value);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Not a valid column type, can't read this column");
  }
  return absl::OkStatus();
}

// Validate that all `Values` have the same expected number of rows.
absl::Status ValidateNumRows(ExampleQueryResult_VectorData_Values values,
                             ColumnSchema::DataType column_type,
                             int expected_num_rows) {
  int num_values;
  switch (column_type) {
    case ColumnSchema::INT32:
      num_values = values.int32_values().value_size();
      break;
    case ColumnSchema::INT64:
      num_values = values.int64_values().value_size();
      break;
    case ColumnSchema::BOOL:
      num_values = values.bool_values().value_size();
      break;
    case ColumnSchema::FLOAT:
      num_values = values.float_values().value_size();
      break;
    case ColumnSchema::DOUBLE:
      num_values = values.double_values().value_size();
      break;
    case ColumnSchema::STRING:
      num_values = values.string_values().value_size();
      break;
    case ColumnSchema::BYTES:
      num_values = values.bytes_values().value_size();
      break;
    default:
      return absl::InvalidArgumentError("Unsupported column type.");
  }
  if (num_values != expected_num_rows) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Column has the wrong number of rows: expected %d but got %d.",
        expected_num_rows, num_values));
  }
  return absl::OkStatus();
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

absl::StatusOr<std::unique_ptr<SqliteAdapter>> SqliteAdapter::Create() {
  int init_result = sqlite3_initialize();
  if (init_result != SQLITE_OK) {
    return absl::InternalError(
        absl::StrFormat("Unable to initialize SQLite library: %s",
                        sqlite3_errstr(init_result)));
  }

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

SqliteAdapter::~SqliteAdapter() {
  sqlite3_close(db_);
  sqlite3_shutdown();
}

absl::Status SqliteAdapter::DefineTable(absl::string_view create_table_stmt) {
  sqlite3_stmt* stmt;
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(
      sqlite3_prepare_v2(db_, create_table_stmt.data(),
                         create_table_stmt.size(), &stmt, nullptr)));
  StatementFinalizer finalizer(stmt);

  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_step(stmt)));

  return absl::OkStatus();
}

absl::Status SqliteAdapter::SetTableContents(TableSchema schema,
                                             SqlData contents) {
  if (contents.vector_data().vectors().size() != schema.column_size()) {
    return absl::InvalidArgumentError(
        "`schema` and `contents` don't have the same number of columns");
  }

  // First clear existing contents from the table
  std::string delete_stmt = absl::StrFormat("DELETE FROM %s;", schema.name());
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(
      sqlite3_exec(db_, delete_stmt.data(), nullptr, nullptr, nullptr)));

  // Insert each row into the table, using parameterized query syntax:
  // INSERT INTO t (c1, c2, ...) VALUES (?, ?, ?);
  std::vector<std::string> column_names(schema.column_size());
  for (int i = 0; i < schema.column_size(); ++i) {
    std::string column_name = schema.column(i).name();
    if (!contents.vector_data().vectors().contains(column_name)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Column `%s` is missing from `contents`.", column_name));
    }
    column_names[i] = column_name;
    FCP_RETURN_IF_ERROR(
        ValidateNumRows(contents.vector_data().vectors().at(column_name),
                        schema.column(i).type(), contents.num_rows()));
  }

  std::string insert_stmt = absl::StrFormat(
      "INSERT INTO %s (%s) VALUES (%s);", schema.name(),
      absl::StrJoin(column_names, ", "),
      absl::StrJoin(std::vector<std::string>(schema.column_size(), "?"), ", "));

  LOG(INFO) << "Insert SQL statement: " << insert_stmt;
  sqlite3_stmt* stmt;
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_prepare_v2(
      db_, insert_stmt.data(), insert_stmt.size(), &stmt, nullptr)));
  StatementFinalizer finalizer(stmt);

  for (int row_num = 0; row_num < contents.num_rows(); ++row_num) {
    for (int col_num = 0; col_num < schema.column_size(); ++col_num) {
      std::string column_name = column_names[col_num];
      ExampleQueryResult_VectorData_Values column_values =
          contents.vector_data().vectors().at(column_name);
      FCP_RETURN_IF_ERROR(
          BindSqliteParameter(stmt, col_num + 1, row_num, column_values,
                              schema.column(col_num).type(), result_handler_));
    }
    FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_step(stmt)));
    FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_reset(stmt)));
    FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_clear_bindings(stmt)));
  }

  return absl::OkStatus();
}

absl::StatusOr<SqlData> SqliteAdapter::EvaluateQuery(
    absl::string_view query, TableSchema output_schema) const {
  SqlData result;
  ExampleQueryResult_VectorData* result_vectors = result.mutable_vector_data();
  for (auto& column : output_schema.column()) {
    (*result_vectors->mutable_vectors())[column.name()];
  }

  sqlite3_stmt* stmt;
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(
      sqlite3_prepare_v2(db_, query.data(), query.size(), &stmt, nullptr)));
  StatementFinalizer finalizer(stmt);
  FCP_RETURN_IF_ERROR(ValidateQueryOutputSchema(stmt, output_schema));

  // SQLite uses `sqlite3_step()` to iterate over result rows, and
  // `sqlite_column_*()` functions to extract values for each column.
  int num_result_rows = 0;
  while (true) {
    int step_result = sqlite3_step(stmt);
    FCP_RETURN_IF_ERROR(result_handler_.ToStatus(step_result));
    if (step_result != SQLITE_ROW) {
      break;
    }
    num_result_rows += 1;
    for (int i = 0; i < output_schema.column_size(); ++i) {
      if (sqlite3_column_type(stmt, i) == SQLITE_NULL) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Encountered NULL value for column `%s`in the query result. "
            "SQLite adapter does not support NULL client result values.",
            output_schema.column(i).name()));
      }
      ColumnSchema::DataType column_type = output_schema.column(i).type();
      std::string column_name = output_schema.column(i).name();
      FCP_RETURN_IF_ERROR(ReadSqliteColumn(
          stmt, column_type, i, result_handler_,
          &result_vectors->mutable_vectors()->at(column_name)));
    }
  }
  result.set_num_rows(num_result_rows);

  return result;
}

}  // namespace confidential_federated_compute::sql_server
