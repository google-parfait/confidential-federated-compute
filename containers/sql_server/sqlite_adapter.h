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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SERVER_SQLITE_ADAPTER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SERVER_SQLITE_ADAPTER_H_

#include <memory>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/sql_server/sql_data.pb.h"
#include "fcp/client/example_query_result.pb.h"
#include "sqlite3.h"

namespace confidential_federated_compute::sql_server {

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

  // Adds a table to the SQLite database context. `create_table_stmt` must be
  // a valid `CREATE TABLE` SQLite statement.
  absl::Status DefineTable(absl::string_view create_table_stmt);

  // Clears contents from the given `table` and inserts the specified
  // `contents`. The table must be created first via `DefineTable`.
  absl::Status SetTableContents(sql_data::TableSchema schema,
                                sql_data::SqlData contents);

  // Evaluates the given SQL `query` statement, producing results in the
  // provided `output_schema`.
  absl::StatusOr<sql_data::SqlData> EvaluateQuery(
      absl::string_view query, sql_data::TableSchema output_schema) const;

 private:
  // The `db` must be non-null. This object takes ownership of the databaseß
  // and releases it within the destructor.
  explicit SqliteAdapter(ABSL_ATTRIBUTE_LIFETIME_BOUND sqlite3* db)
      : result_handler_(db), db_(db) {}

  SqliteResultHandler result_handler_;
  sqlite3* db_;
};

}  // namespace confidential_federated_compute::sql_server

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SERVER_SQLITE_ADAPTER_H_
