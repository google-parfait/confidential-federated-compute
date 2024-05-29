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

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "fcp/protos/plan.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "sqlite3.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sql_server {

using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::TableSchema;
using ::google::internal::federated::plan::ExampleQuerySpec;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_BOOL;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_BYTES;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_DOUBLE;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_FLOAT;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_INT32;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_INT64;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_STRING;
using ::google::internal::federated::plan::Plan;
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
// row_num: The index of the value to bind in `column_values`.
//
// column_values: A `Values` representing a column.
//
// column_type: The data type of `column_values`.
//
// status_util: Utility for inspecting SQLite result codes and translating them
// `absl::Status`.
absl::Status BindSqliteParameter(sqlite3_stmt* stmt, int ordinal, int row_num,
                                 const TensorColumn& column,
                                 const SqliteResultHandler& status_util) {
  switch (column.column_schema_.type()) {
    case ExampleQuerySpec_OutputVectorSpec_DataType_INT32:
      return status_util.ToStatus(sqlite3_bind_int(
          stmt, ordinal, column.tensor_.AsSpan<int32_t>().at(row_num)));
    case ExampleQuerySpec_OutputVectorSpec_DataType_INT64:
      return status_util.ToStatus(sqlite3_bind_int64(
          stmt, ordinal, column.tensor_.AsSpan<int64_t>().at(row_num)));
    case ExampleQuerySpec_OutputVectorSpec_DataType_BOOL:
      return status_util.ToStatus(sqlite3_bind_int(
          stmt, ordinal, column.tensor_.AsSpan<int32_t>().at(row_num)));
    case ExampleQuerySpec_OutputVectorSpec_DataType_FLOAT:
      return status_util.ToStatus(sqlite3_bind_double(
          stmt, ordinal, column.tensor_.AsSpan<float>().at(row_num)));
    case ExampleQuerySpec_OutputVectorSpec_DataType_DOUBLE:
      return status_util.ToStatus(sqlite3_bind_double(
          stmt, ordinal, column.tensor_.AsSpan<double>().at(row_num)));
    case ExampleQuerySpec_OutputVectorSpec_DataType_BYTES: {
      absl::string_view string_value =
          column.tensor_.AsSpan<absl::string_view>().at(row_num);
      return status_util.ToStatus(
          sqlite3_bind_blob(stmt, ordinal, string_value.data(),
                            string_value.size(), SQLITE_TRANSIENT));
    }
    case ExampleQuerySpec_OutputVectorSpec_DataType_STRING: {
      absl::string_view string_value =
          column.tensor_.AsSpan<absl::string_view>().at(row_num);
      // Using sqlite3_bind_text means the string should be well-formed UTF8
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
    sqlite3_stmt* stmt, ExampleQuerySpec_OutputVectorSpec_DataType column_type,
    int column_index, const SqliteResultHandler& status_util,
    TensorData* result) {
  switch (column_type) {
    case ExampleQuerySpec_OutputVectorSpec_DataType_INT32:
      StoreValue<int32_t>(result, sqlite3_column_int(stmt, column_index));
      break;
    case ExampleQuerySpec_OutputVectorSpec_DataType_INT64:
      StoreValue<int64_t>(result, sqlite3_column_int64(stmt, column_index));
      break;
    case ExampleQuerySpec_OutputVectorSpec_DataType_BOOL:
      StoreValue<int32_t>(result, sqlite3_column_int(stmt, column_index) == 1);
      break;
    case ExampleQuerySpec_OutputVectorSpec_DataType_FLOAT:
      StoreValue<float>(result, sqlite3_column_double(stmt, column_index));
      break;
    case ExampleQuerySpec_OutputVectorSpec_DataType_DOUBLE:
      StoreValue<double>(result, sqlite3_column_double(stmt, column_index));
      break;
    case ExampleQuerySpec_OutputVectorSpec_DataType_BYTES: {
      std::string bytes_value(
          static_cast<const char*>(sqlite3_column_blob(stmt, column_index)),
          sqlite3_column_bytes(stmt, column_index));
      StoreValue<std::string>(result, std::move(bytes_value));
      break;
    }
    case ExampleQuerySpec_OutputVectorSpec_DataType_STRING: {
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
    ExampleQuerySpec_OutputVectorSpec_DataType sql_type) {
  switch (sql_type) {
    case ExampleQuerySpec_OutputVectorSpec_DataType_INT32:
      return DataType::DT_INT32;
    case ExampleQuerySpec_OutputVectorSpec_DataType_INT64:
      return DataType::DT_INT64;
    case ExampleQuerySpec_OutputVectorSpec_DataType_BOOL:
      return DataType::DT_INT32;
    case ExampleQuerySpec_OutputVectorSpec_DataType_FLOAT:
      return DataType::DT_FLOAT;
    case ExampleQuerySpec_OutputVectorSpec_DataType_DOUBLE:
      return DataType::DT_DOUBLE;
    case ExampleQuerySpec_OutputVectorSpec_DataType_STRING:
    case ExampleQuerySpec_OutputVectorSpec_DataType_BYTES:
      return DataType::DT_STRING;
    default:
      return absl::InvalidArgumentError("Unsupported column type.");
  }
}

absl::Status ValidateInputColumns(const std::vector<TensorColumn>& columns,
                                  int num_rows) {
  for (const TensorColumn& column : columns) {
    if (column.tensor_.num_elements() != num_rows) {
      return absl::InvalidArgumentError("Column has the wrong number of rows");
    }
  }
  return absl::OkStatus();
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

absl::StatusOr<TensorColumn> TensorColumn::Create(ColumnSchema column_schema,
                                                  Tensor tensor) {
  FCP_ASSIGN_OR_RETURN(DataType schema_dtype,
                       SqlDataTypeToTensorDtype(column_schema.type()));
  if (tensor.dtype() != schema_dtype) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Column `%s` type (%s) does not match the ColumnSchema type (%s).",
        column_schema.name(), DataType_Name(tensor.dtype()),
        ExampleQuerySpec::OutputVectorSpec::DataType_Name(
            column_schema.type())));
  }
  return TensorColumn(column_schema, std::move(tensor));
}

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

  std::vector<std::string> column_names(schema.column_size());
  for (int i = 0; i < schema.column_size(); ++i) {
    column_names[i] = schema.column(i).name();
  }

  insert_stmt_ = absl::StrFormat(
      "INSERT INTO %s (%s) VALUES (%s);", schema.name(),
      absl::StrJoin(column_names, ", ", &EscapeSqlColumnName),
      absl::StrJoin(std::vector<std::string>(schema.column_size(), "?"), ", "));

  return absl::OkStatus();
}

absl::Status SqliteAdapter::AddTableContents(std::vector<TensorColumn> contents,
                                             int num_rows) {
  if (!insert_stmt_.has_value()) {
    return absl::InvalidArgumentError(
        "`DefineTable` must be called before adding to the table contents.");
  }
  FCP_RETURN_IF_ERROR(ValidateInputColumns(contents, num_rows));
  // Insert each row into the table, using parameterized query syntax:
  // INSERT INTO t (c1, c2, ...) VALUES (?, ?, ?);
  sqlite3_stmt* stmt;
  FCP_RETURN_IF_ERROR(result_handler_.ToStatus(
      sqlite3_prepare_v2(db_, insert_stmt_.value().data(),
                         insert_stmt_.value().size(), &stmt, nullptr)));
  StatementFinalizer finalizer(stmt);

  for (int row_num = 0; row_num < num_rows; ++row_num) {
    for (int col_num = 0; col_num < contents.size(); ++col_num) {
      FCP_RETURN_IF_ERROR(BindSqliteParameter(
          stmt, col_num + 1, row_num, contents.at(col_num), result_handler_));
    }
    FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_step(stmt)));
    FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_reset(stmt)));
    FCP_RETURN_IF_ERROR(result_handler_.ToStatus(sqlite3_clear_bindings(stmt)));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<TensorColumn>> SqliteAdapter::EvaluateQuery(
    absl::string_view query,
    const RepeatedPtrField<ColumnSchema>& output_columns) const {
  std::vector<std::unique_ptr<TensorData>> result_columns(
      output_columns.size());
  for (int i = 0; i < output_columns.size(); i++) {
    std::unique_ptr<TensorData> data;
    switch (output_columns.at(i).type()) {
      case ExampleQuerySpec_OutputVectorSpec_DataType_INT32:
        data = std::make_unique<MutableVectorData<int32_t>>();
        break;
      case ExampleQuerySpec_OutputVectorSpec_DataType_INT64:
        data = std::make_unique<MutableVectorData<int64_t>>();
        break;
      case ExampleQuerySpec_OutputVectorSpec_DataType_BOOL:
        data = std::make_unique<MutableVectorData<int32_t>>();
        break;
      case ExampleQuerySpec_OutputVectorSpec_DataType_FLOAT:
        data = std::make_unique<MutableVectorData<float>>();
        break;
      case ExampleQuerySpec_OutputVectorSpec_DataType_DOUBLE:
        data = std::make_unique<MutableVectorData<double>>();
        break;
      case ExampleQuerySpec_OutputVectorSpec_DataType_STRING:
      case ExampleQuerySpec_OutputVectorSpec_DataType_BYTES:
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
  std::vector<TensorColumn> result(output_columns.size());

  for (int i = 0; i < output_columns.size(); ++i) {
    FCP_ASSIGN_OR_RETURN(DataType dtype,
                         SqlDataTypeToTensorDtype(output_columns.at(i).type()));
    FCP_ASSIGN_OR_RETURN(
        Tensor column_tensor,
        Tensor::Create(dtype, {num_rows}, std::move(result_columns.at(i))));
    FCP_ASSIGN_OR_RETURN(
        TensorColumn tensor_column,
        TensorColumn::Create(output_columns.at(i), std::move(column_tensor)));
    result[i] = std::move(tensor_column);
  }
  return result;
}

}  // namespace confidential_federated_compute::sql_server
