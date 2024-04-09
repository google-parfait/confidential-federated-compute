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

#include <thread>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::sql_server {

namespace {

using ::fcp::aggregation::CreateTestData;
using ::fcp::aggregation::DataType;
using ::fcp::aggregation::Tensor;
using ::fcp::aggregation::TensorData;
using ::fcp::aggregation::TensorShape;
using ::fcp::client::ExampleQueryResult_VectorData;
using ::fcp::client::ExampleQueryResult_VectorData_Int64Values;
using ::fcp::client::ExampleQueryResult_VectorData_StringValues;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::TableSchema;
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
using ::testing::HasSubstr;
using ::testing::Test;

TEST(TensorColumnTest, ValidCreate) {
  absl::StatusOr<Tensor> int_tensor = Tensor::Create(
      DataType::DT_INT64, {1}, std::move(CreateTestData<uint64_t>({1})));
  CHECK_OK(int_tensor);

  ColumnSchema int_col_schema;
  int_col_schema.set_name("col");
  int_col_schema.set_type(ExampleQuerySpec_OutputVectorSpec_DataType_INT64);

  absl::StatusOr<TensorColumn> int_column =
      TensorColumn::Create(int_col_schema, std::move(int_tensor.value()));
  CHECK_OK(int_column);
}

TEST(TensorColumnTest, InvalidCreate) {
  absl::StatusOr<Tensor> int_tensor = Tensor::Create(
      DataType::DT_INT64, {1}, std::move(CreateTestData<uint64_t>({1})));
  CHECK_OK(int_tensor);

  ColumnSchema str_col_schema;
  str_col_schema.set_name("col");
  str_col_schema.set_type(ExampleQuerySpec_OutputVectorSpec_DataType_STRING);

  absl::StatusOr<TensorColumn> int_column =
      TensorColumn::Create(str_col_schema, std::move(int_tensor.value()));

  ASSERT_TRUE(absl::IsInvalidArgument(int_column.status()));
  ASSERT_THAT(int_column.status().message(),
              HasSubstr("Column `col` type (DT_INT64) does not match the "
                        "ColumnSchema type (STRING)"));
}

class SqliteAdapterTest : public Test {
 protected:
  std::unique_ptr<SqliteAdapter> sqlite_;

  void SetUp() override {
    CHECK_OK(SqliteAdapter::Initialize());
    absl::StatusOr<std::unique_ptr<SqliteAdapter>> create_status =
        SqliteAdapter::Create();
    CHECK_OK(create_status);
    sqlite_ = std::move(create_status.value());
  }

  void TearDown() override {
    sqlite_.reset();
    SqliteAdapter::ShutDown();
  }
};

TableSchema CreateInputTableSchema(absl::string_view table_name = "t",
                                   absl::string_view col1_name = "int_vals",
                                   absl::string_view col2_name = "str_vals") {
  TableSchema schema;
  schema.set_name(std::string(table_name));
  ColumnSchema* col1 = schema.add_column();
  col1->set_name(std::string(col1_name));
  col1->set_type(ExampleQuerySpec_OutputVectorSpec_DataType_INT64);
  ColumnSchema* col2 = schema.add_column();
  col2->set_name(std::string(col2_name));
  col2->set_type(ExampleQuerySpec_OutputVectorSpec_DataType_STRING);
  const std::string create_table_stmt =
      absl::StrFormat(R"sql(CREATE TABLE %s (%s INTEGER, %s TEXT))sql",
                      table_name, col1_name, col2_name);
  schema.set_create_table_sql(std::string(create_table_stmt));

  return schema;
}

TEST(MultipleThreadsTest, ConcurrentDefineTable) {
  CHECK_OK(SqliteAdapter::Initialize());
  std::thread thread1([]() {
    absl::StatusOr<std::unique_ptr<SqliteAdapter>> create_status =
        SqliteAdapter::Create();
    CHECK_OK(create_status);
    std::unique_ptr<SqliteAdapter> adapter = std::move(create_status.value());
    CHECK_OK(adapter->DefineTable(CreateInputTableSchema("table_1")));
  });
  std::thread thread2([]() {
    absl::StatusOr<std::unique_ptr<SqliteAdapter>> create_status =
        SqliteAdapter::Create();
    CHECK_OK(create_status);
    std::unique_ptr<SqliteAdapter> adapter = std::move(create_status.value());
    CHECK_OK(adapter->DefineTable(CreateInputTableSchema("table_2")));
  });
  thread1.join();
  thread2.join();
  SqliteAdapter::ShutDown();
}

class DefineTableTest : public SqliteAdapterTest {};

TEST_F(DefineTableTest, ValidCreateTableStatement) {
  ASSERT_TRUE(sqlite_->DefineTable(CreateInputTableSchema()).ok());
}

TEST_F(DefineTableTest, InvalidCreateTableStatement) {
  TableSchema schema;
  schema.set_create_table_sql("BAD STATEMENT blah blah");
  absl::Status result_status = sqlite_->DefineTable(schema);
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(), HasSubstr("syntax error"));
}

class SetTableContentsTest : public SqliteAdapterTest {
 protected:
  absl::StatusOr<std::vector<TensorColumn>> CreateTableContents(
      std::initializer_list<uint64_t> int_vals,
      std::initializer_list<absl::string_view> str_vals,
      absl::string_view int_col_name = "int_vals",
      absl::string_view str_col_name = "str_vals") {
    std::vector<TensorColumn> contents;

    FCP_ASSIGN_OR_RETURN(
        Tensor int_tensor,
        Tensor::Create(DataType::DT_INT64,
                       {static_cast<int64_t>(int_vals.size())},
                       std::move(CreateTestData<uint64_t>(int_vals))));
    ColumnSchema int_col_schema;
    int_col_schema.set_name(std::string(int_col_name));
    int_col_schema.set_type(ExampleQuerySpec_OutputVectorSpec_DataType_INT64);

    FCP_ASSIGN_OR_RETURN(
        Tensor str_tensor,
        Tensor::Create(DataType::DT_STRING,
                       {static_cast<int64_t>(str_vals.size())},
                       std::move(CreateTestData<absl::string_view>(str_vals))));
    ColumnSchema str_col_schema;
    str_col_schema.set_name(std::string(str_col_name));
    str_col_schema.set_type(ExampleQuerySpec_OutputVectorSpec_DataType_STRING);

    FCP_ASSIGN_OR_RETURN(
        TensorColumn int_column,
        TensorColumn::Create(int_col_schema, std::move(int_tensor)));
    FCP_ASSIGN_OR_RETURN(
        TensorColumn str_column,
        TensorColumn::Create(str_col_schema, std::move(str_tensor)));
    contents.push_back(std::move(int_column));
    contents.push_back(std::move(str_column));
    return contents;
  }
};

TEST_F(SetTableContentsTest, BasicUsage) {
  CHECK_OK(sqlite_->DefineTable(CreateInputTableSchema()));
  absl::StatusOr<std::vector<TensorColumn>> contents =
      CreateTableContents({1, 2, 3}, {"a", "b", "c"});
  CHECK_OK(contents);

  CHECK_OK(sqlite_->AddTableContents(std::move(contents.value()), 3));
}

TEST_F(SetTableContentsTest, ColumnNameEscaping) {
  TableSchema schema = CreateInputTableSchema(
      /*table_name=*/"t", /*col1_name=*/"![]^'\";/int_vals",
      /*col2_name=*/"![]^'\";/str_vals");
  // CreateInputTableSchema doesn't properly escape create_table_sql.
  schema.set_create_table_sql(
      "CREATE TABLE t ('![]^''\";/int_vals' INTEGER, '![]^''\";/str_vals' "
      "TEXT)");

  CHECK_OK(sqlite_->DefineTable(schema));
  absl::StatusOr<std::vector<TensorColumn>> contents =
      CreateTableContents({1, 2, 3}, {"a", "b", "c"}, schema.column(0).name(),
                          schema.column(1).name());
  CHECK_OK(contents);

  CHECK_OK(sqlite_->AddTableContents(std::move(contents.value()), 3));
}

TEST_F(SetTableContentsTest, CalledBeforeDefineTable) {
  absl::StatusOr<std::vector<TensorColumn>> contents =
      CreateTableContents({1}, {"a"});
  CHECK_OK(contents);

  absl::Status result_status =
      sqlite_->AddTableContents(std::move(contents.value()), 5);
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(),
              HasSubstr("`DefineTable` must be called before"));
}

TEST_F(SetTableContentsTest, NumRowsTooLarge) {
  CHECK_OK(sqlite_->DefineTable(CreateInputTableSchema()));
  absl::StatusOr<std::vector<TensorColumn>> contents =
      CreateTableContents({1, 2, 3}, {"a", "b", "c"});
  CHECK_OK(contents);

  absl::Status result_status =
      sqlite_->AddTableContents(std::move(contents.value()), 5);
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(),
              HasSubstr("Column has the wrong number of rows"));
}

TEST_F(SetTableContentsTest, NumRowsTooSmall) {
  CHECK_OK(sqlite_->DefineTable(CreateInputTableSchema()));
  absl::StatusOr<std::vector<TensorColumn>> contents =
      CreateTableContents({1, 2, 3}, {"a", "b", "c"});
  CHECK_OK(contents);

  absl::Status result_status =
      sqlite_->AddTableContents(std::move(contents.value()), 1);
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(),
              HasSubstr("Column has the wrong number of rows"));
}

TEST_F(SetTableContentsTest, ZeroRows) {
  CHECK_OK(sqlite_->DefineTable(CreateInputTableSchema()));
  absl::StatusOr<std::vector<TensorColumn>> contents =
      CreateTableContents({}, {});
  TableSchema schema = CreateInputTableSchema();
  CHECK_OK(contents);

  CHECK_OK(sqlite_->AddTableContents(std::move(contents.value()), 0));
}

class EvaluateQueryTest : public SetTableContentsTest {
 protected:
  void SetUp() override {
    SetTableContentsTest::SetUp();
    CHECK_OK(sqlite_->DefineTable(CreateInputTableSchema()));
  }

  void SetColumnNameAndType(ColumnSchema* col, std::string name,
                            ExampleQuerySpec_OutputVectorSpec_DataType type) {
    col->set_name(name);
    col->set_type(type);
  }
};

TEST_F(EvaluateQueryTest, ValidQueryBasicExpression) {
  const std::string query = R"sql(
    SELECT 1+1 AS two
  )sql";
  std::string output_col_name = "two";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       ExampleQuerySpec_OutputVectorSpec_DataType_INT64);

  auto result_status = sqlite_->EvaluateQuery(query, output_schema.column());
  ASSERT_TRUE(result_status.ok());
  std::vector<TensorColumn> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).tensor_.dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).tensor_.num_elements(), 1);
  ASSERT_EQ(result.at(0).tensor_.AsSpan<int64_t>().at(0), 2);
}

TEST_F(EvaluateQueryTest, ValidQueryBasicStringExpression) {
  const std::string query = R"sql(
    SELECT "foo" AS str_val
  )sql";
  std::string output_col_name = "str_val";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       ExampleQuerySpec_OutputVectorSpec_DataType_STRING);

  auto result_status = sqlite_->EvaluateQuery(query, output_schema.column());
  ASSERT_TRUE(result_status.ok());
  std::vector<TensorColumn> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).tensor_.dtype(), DataType::DT_STRING);
  ASSERT_EQ(result.at(0).tensor_.num_elements(), 1);
  ASSERT_EQ(result.at(0).tensor_.AsSpan<absl::string_view>().at(0), "foo");
}

TEST_F(EvaluateQueryTest, ValidQueryScalarFunction) {
  const std::string query = R"sql(
    SELECT ABS(-2) AS two
  )sql";
  std::string output_col_name = "two";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       ExampleQuerySpec_OutputVectorSpec_DataType_INT64);

  auto result_status = sqlite_->EvaluateQuery(query, output_schema.column());
  ASSERT_TRUE(result_status.ok());
  std::vector<TensorColumn> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).tensor_.dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).tensor_.num_elements(), 1);
  ASSERT_EQ(result.at(0).tensor_.AsSpan<int64_t>().at(0), 2);
}

TEST_F(EvaluateQueryTest, InvalidQueryParseError) {
  const std::string query = R"sql(
    BAD STATEMENT blah blah blah
  )sql";
  TableSchema empty_schema;

  auto result = sqlite_->EvaluateQuery(query, empty_schema.column());
  ASSERT_TRUE(absl::IsInvalidArgument(result.status()));
  ASSERT_THAT(result.status().message(), HasSubstr("syntax error"));
}

TEST_F(EvaluateQueryTest, InvalidQueryMissingFunction) {
  // `IF()` function is available in GoogleSQL but not SQLite.
  const std::string query = R"sql(
    SELECT IF(TRUE, 1, 0) AS one
  )sql";
  TableSchema empty_schema;

  auto result = sqlite_->EvaluateQuery(query, empty_schema.column());
  ASSERT_TRUE(absl::IsInvalidArgument(result.status()));
  ASSERT_THAT(result.status().message(), HasSubstr("no such function: IF"));
}

TEST_F(EvaluateQueryTest, InvalidQueryInvalidTable) {
  const std::string query = R"sql(
    SELECT val FROM missing_table
  )sql";
  TableSchema empty_schema;

  auto result = sqlite_->EvaluateQuery(query, empty_schema.column());
  ASSERT_TRUE(absl::IsInvalidArgument(result.status()));
  ASSERT_THAT(result.status().message(),
              HasSubstr("no such table: missing_table"));
}

TEST_F(EvaluateQueryTest, InvalidQueryInvalidColumn) {
  const std::string query = R"sql(
    SELECT missing_col FROM t
  )sql";
  TableSchema empty_schema;

  auto result = sqlite_->EvaluateQuery(query, empty_schema.column());
  ASSERT_TRUE(absl::IsInvalidArgument(result.status()));
  ASSERT_THAT(result.status().message(),
              HasSubstr("no such column: missing_col"));
}

TEST_F(EvaluateQueryTest, EmptyResults) {
  std::string output_col_name = "x";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       ExampleQuerySpec_OutputVectorSpec_DataType_INT64);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(
                            WITH D (x) AS (VALUES (1), (2), (3))
                            SELECT * FROM D WHERE FALSE;
                           )sql",
      output_schema.column());

  ASSERT_TRUE(result_status.ok());
  std::vector<TensorColumn> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).tensor_.dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).tensor_.num_elements(), 0);
}

TEST_F(EvaluateQueryTest, ResultsFromTable) {
  absl::StatusOr<std::vector<TensorColumn>> contents =
      CreateTableContents({42}, {"a"});
  CHECK_OK(contents);

  CHECK_OK(sqlite_->AddTableContents(std::move(contents.value()), 1));

  std::string output_col_name = "int_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       ExampleQuerySpec_OutputVectorSpec_DataType_INT64);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(SELECT int_vals FROM t;)sql", output_schema.column());
  ASSERT_TRUE(result_status.ok());
  std::vector<TensorColumn> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).tensor_.dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).tensor_.num_elements(), 1);
  ASSERT_EQ(result.at(0).tensor_.AsSpan<int64_t>().at(0), 42);
}

TEST_F(EvaluateQueryTest, MultipleSetTableContents) {
  int num_rows = 3;
  int kNumSetContents = 5;
  for (int i = 0; i < kNumSetContents; ++i) {
    absl::StatusOr<std::vector<TensorColumn>> contents =
        CreateTableContents({1, 2, 3}, {"a", "b", "c"});
    CHECK_OK(contents);
    CHECK_OK(sqlite_->AddTableContents(std::move(contents.value()), num_rows));
  }

  std::string output_col_name = "n";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       ExampleQuerySpec_OutputVectorSpec_DataType_INT64);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(SELECT COUNT(*) AS n FROM t;)sql", output_schema.column());
  ASSERT_TRUE(result_status.ok());
  std::vector<TensorColumn> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).tensor_.dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).tensor_.num_elements(), 1);
  ASSERT_EQ(result.at(0).tensor_.AsSpan<int64_t>().at(0),
            num_rows * kNumSetContents);
}

TEST_F(EvaluateQueryTest, MultipleResultRows) {
  absl::StatusOr<std::vector<TensorColumn>> contents =
      CreateTableContents({3, 2}, {"a", "b"});
  CHECK_OK(contents);
  CHECK_OK(sqlite_->AddTableContents(std::move(contents.value()), 2));

  std::string output_col_name = "int_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       ExampleQuerySpec_OutputVectorSpec_DataType_INT64);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(SELECT int_vals FROM t ORDER BY int_vals;)sql",
      output_schema.column());

  ASSERT_TRUE(result_status.ok());
  std::vector<TensorColumn> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).tensor_.dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).tensor_.num_elements(), 2);
  ASSERT_EQ(result.at(0).tensor_.AsSpan<int64_t>().at(0), 2);
  ASSERT_EQ(result.at(0).tensor_.AsSpan<int64_t>().at(1), 3);
}

TEST_F(EvaluateQueryTest, MultipleColumns) {
  absl::StatusOr<std::vector<TensorColumn>> contents =
      CreateTableContents({1}, {"a"});
  CHECK_OK(contents);
  CHECK_OK(sqlite_->AddTableContents(std::move(contents.value()), 1));

  std::string int_output_col_name = "int_vals";
  std::string str_output_col_name = "str_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), int_output_col_name,
                       ExampleQuerySpec_OutputVectorSpec_DataType_INT64);
  SetColumnNameAndType(output_schema.add_column(), str_output_col_name,
                       ExampleQuerySpec_OutputVectorSpec_DataType_STRING);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(SELECT int_vals, str_vals FROM t;)sql", output_schema.column());

  ASSERT_TRUE(result_status.ok());
  std::vector<TensorColumn> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result.at(0).tensor_.dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).tensor_.num_elements(), 1);
  ASSERT_EQ(result.at(0).tensor_.AsSpan<int64_t>().at(0), 1);
  ASSERT_EQ(result.at(1).tensor_.dtype(), DataType::DT_STRING);
  ASSERT_EQ(result.at(1).tensor_.num_elements(), 1);
  ASSERT_EQ(result.at(1).tensor_.AsSpan<absl::string_view>().at(0), "a");
}

TEST_F(EvaluateQueryTest, IncorrectSchemaNumColumns) {
  std::string output_col_name = "v1";
  TableSchema output_schema;
  ColumnSchema* column = output_schema.add_column();
  column->set_name(output_col_name);
  column->set_type(ExampleQuerySpec_OutputVectorSpec_DataType_INT64);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(
                    WITH D (v1, v2) AS (VALUES (1, 2), (3, 4), (5, 6))
                    SELECT * FROM D;
                    )sql",
      output_schema.column());

  ASSERT_TRUE(absl::IsInvalidArgument(result_status.status()));
  ASSERT_THAT(
      result_status.status().message(),
      HasSubstr("Query results did not match the specified output columns"));
}

TEST_F(EvaluateQueryTest, IncorrectSchemaColumnNames) {
  std::string output_col_name = "v1";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       ExampleQuerySpec_OutputVectorSpec_DataType_INT64);

  auto result_status =
      sqlite_->EvaluateQuery(R"sql(SELECT 42 AS v)sql", output_schema.column());

  ASSERT_TRUE(absl::IsInvalidArgument(result_status.status()));
  ASSERT_THAT(
      result_status.status().message(),
      HasSubstr("Query results did not match the specified output columns"));
}

TEST_F(EvaluateQueryTest, AllSupportedDataTypes) {
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), "int32_val",
                       ExampleQuerySpec_OutputVectorSpec_DataType_INT32);
  SetColumnNameAndType(output_schema.add_column(), "int64_val",
                       ExampleQuerySpec_OutputVectorSpec_DataType_INT64);
  SetColumnNameAndType(output_schema.add_column(), "float_val",
                       ExampleQuerySpec_OutputVectorSpec_DataType_FLOAT);
  SetColumnNameAndType(output_schema.add_column(), "double_val",
                       ExampleQuerySpec_OutputVectorSpec_DataType_DOUBLE);
  SetColumnNameAndType(output_schema.add_column(), "bool_val",
                       ExampleQuerySpec_OutputVectorSpec_DataType_BOOL);
  SetColumnNameAndType(output_schema.add_column(), "str_val",
                       ExampleQuerySpec_OutputVectorSpec_DataType_STRING);
  SetColumnNameAndType(output_schema.add_column(), "binary_val",
                       ExampleQuerySpec_OutputVectorSpec_DataType_BYTES);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(
                            SELECT
                              1 AS int32_val,
                              1 AS int64_val,
                              1.0 AS float_val,
                              1.0 AS double_val,
                              TRUE AS bool_val,
                              "foobar" AS str_val,
                              "arbitrary bytes" AS binary_val;
                            )sql",
      output_schema.column());

  ASSERT_TRUE(result_status.ok()) << result_status.status().message();
}

}  // namespace

}  // namespace confidential_federated_compute::sql_server
