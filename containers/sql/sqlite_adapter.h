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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SQLITE_ADAPTER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SQLITE_ADAPTER_H_

#include <memory>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "sqlite3.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sql {

// Utility for inspecting SQLite result codes and translating them
// `absl::Status`.
class SqliteResultHandler final {
 public:
  explicit SqliteResultHandler(sqlite3* db ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : db_(db) {}

  // Translates the given `result_code` to an `absl::Status`.
  absl::Status ToStatus(
      int result_code,
      absl::StatusCode error_status = absl::StatusCode::kInvalidArgument) const;

 private:
  sqlite3* db_;
};

// Wrapper for an in-memory SQLite database. Provides higher-level APIs wrapping
// SQLite's C interface. This class is not threasdafe.
class SqliteAdapter {
 public:
  //  SQLite's limit on the number of variables in a single statement.
  static constexpr int kSqliteVariableLimit = 32766;

  //  Initializes the SQLite library.
  static absl::Status Initialize();

  // Creates a new `SqliteAdapter` object. The SQLite library must be
  // initialized with `Initialize` before any adapters are created.
  static absl::StatusOr<std::unique_ptr<SqliteAdapter>> Create();

  // Shuts down the SQLite library. All open database connections must be closed
  // before invoking this.
  static void ShutDown();

  SqliteAdapter(const SqliteAdapter&) = delete;
  SqliteAdapter& operator=(const SqliteAdapter&) = delete;

  // Closes the database connection.
  ~SqliteAdapter();

  // Adds a table to the SQLite database context.
  absl::Status DefineTable(fcp::confidentialcompute::TableSchema schema);

  // Inserts the specified `contents` into the table specified by `table_name`.
  // The table must be created first via the most recent call to `DefineTable`,
  // and the column order of `contents` must match the schema specified for
  // `DefineTable`.
  absl::Status AddTableContents(
      const std::vector<tensorflow_federated::aggregation::Tensor>& contents,
      int num_rows);

  // Evaluates the given SQL `query` statement, producing results matching the
  // provided `output_columns`.
  absl::StatusOr<std::vector<tensorflow_federated::aggregation::Tensor>>
  EvaluateQuery(
      absl::string_view query,
      const google::protobuf::RepeatedPtrField<
          fcp::confidentialcompute::ColumnSchema>& output_columns) const;

 private:
  // The `db` must be non-null. This object takes ownership of the databaseß
  // and releases it within the destructor.
  explicit SqliteAdapter(ABSL_ATTRIBUTE_LIFETIME_BOUND sqlite3* db)
      : result_handler_(db), db_(db) {}

  // Insert rows from `contents` from `start_row_index` (inclusive) to
  // `end_row_index` (exclusive).
  // `insert_stmt` should be of the form
  // "INSERT INTO <table name> (<col name 1>, <col name 2>, ...) VALUES (?, ?,
  // ...), ..." and the columns should match `contents`, and the number of rows
  // should match `end_row_index -  start_row_index`.
  absl::Status InsertRows(
      const std::vector<tensorflow_federated::aggregation::Tensor>& contents,
      int start_row_index, int end_row_index, absl::string_view insert_stmt);

  SqliteResultHandler result_handler_;
  sqlite3* db_;
  std::optional<fcp::confidentialcompute::TableSchema> table_schema_;
};

}  // namespace confidential_federated_compute::sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SQLITE_ADAPTER_H_
