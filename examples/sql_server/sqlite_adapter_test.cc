#include "examples/sql_server/sqlite_adapter.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::fcp::client::ExampleQueryResult_VectorData_Int64Values;
using ::fcp::client::ExampleQueryResult_VectorData_StringValues;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::sql_data::TableSchema;
using ::testing::HasSubstr;
using ::testing::Test;

class SqliteAdapterTest : public Test {
 protected:
  std::unique_ptr<SqliteAdapter> sqlite_;

  void SetUp() override {
    absl::StatusOr<std::unique_ptr<SqliteAdapter>> create_status =
        SqliteAdapter::Create();
    CHECK_OK(create_status);
    sqlite_ = std::move(create_status.value());
  }
};

class DefineTableTest : public SqliteAdapterTest {};

TEST_F(DefineTableTest, ValidCreateTableStatement) {
  const std::string create_table_stmt = R"sql(
    CREATE TABLE foo (col1 TEXT)
  )sql";

  ASSERT_TRUE(sqlite_->DefineTable(create_table_stmt).ok());
}

TEST_F(DefineTableTest, InvalidCreateTableStatement) {
  const std::string bad_create_table_stmt = R"sql(
    BAD STATEMENT blah blah blah
  )sql";

  absl::Status result_status = sqlite_->DefineTable(bad_create_table_stmt);
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(), HasSubstr("syntax error"));
}

class SetTableContentsTest : public SqliteAdapterTest {
 protected:
  static constexpr absl::string_view kTableName = "t";

  void SetUp() override {
    SqliteAdapterTest::SetUp();
    CHECK_OK(sqlite_->DefineTable(absl::StrFormat(
        R"sql(
          CREATE TABLE %s (int_vals INTEGER, str_vals TEXT);
        )sql",
        kTableName)));
  }

  ExampleQueryResult_VectorData_Values CreateInt64Values(
      std::initializer_list<int> input_values) {
    ExampleQueryResult_VectorData_Values values;
    ExampleQueryResult_VectorData_Int64Values* int_values =
        values.mutable_int64_values();
    for (int i : input_values) {
      int_values->add_value(i);
    }
    return values;
  }

  ExampleQueryResult_VectorData_Values CreateStringValues(
      std::initializer_list<std::string> input_values) {
    ExampleQueryResult_VectorData_Values values;
    ExampleQueryResult_VectorData_StringValues* string_values =
        values.mutable_string_values();
    for (std::string i : input_values) {
      string_values->add_value(i);
    }
    return values;
  }

  absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
  CreateTableContents(std::initializer_list<int> int_vals,
                      std::initializer_list<std::string> str_vals) {
    absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
        contents;
    contents["int_vals"] = CreateInt64Values(int_vals);
    contents["str_vals"] = CreateStringValues(str_vals);
    return contents;
  }

  TableSchema CreateInputTableSchema(absl::string_view table_name = kTableName,
                                     absl::string_view col1_name = "int_vals",
                                     absl::string_view col2_name = "str_vals") {
    TableSchema schema;
    schema.set_name(std::string(table_name));
    ColumnSchema* col1 = schema.add_column();
    col1->set_name(std::string(col1_name));
    col1->set_type(ColumnSchema::DataType::ColumnSchema_DataType_INT64);
    ColumnSchema* col2 = schema.add_column();
    col2->set_name(std::string(col2_name));
    col2->set_type(ColumnSchema::DataType::ColumnSchema_DataType_STRING);
    return schema;
  }
};

TEST_F(SetTableContentsTest, BasicUsage) {
  absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
      contents = CreateTableContents({1, 2, 3}, {"a", "b", "c"});
  TableSchema schema = CreateInputTableSchema();
  int num_rows = 3;

  CHECK_OK(sqlite_->SetTableContents(schema, contents, num_rows));
}

TEST_F(SetTableContentsTest, TableDoesNotExist) {
  absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
      contents = CreateTableContents({1}, {"a"});
  TableSchema schema = CreateInputTableSchema("nonexistent");
  int num_rows = 1;

  absl::Status result_status =
      sqlite_->SetTableContents(schema, contents, num_rows);
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(), HasSubstr("no such table"));
}

TEST_F(SetTableContentsTest, ExtraColumn) {
  absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
      contents = CreateTableContents({1}, {"a"});
  TableSchema schema = CreateInputTableSchema(kTableName, "extra_col");
  int num_rows = 1;

  absl::Status result_status =
      sqlite_->SetTableContents(schema, contents, num_rows);
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(),
              HasSubstr("table t has no column named extra_col"));
}

TEST_F(SetTableContentsTest, MissingColumn) {
  absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
      contents;
  contents["int_vals"] = CreateInt64Values({1, 2, 3});
  TableSchema schema = CreateInputTableSchema();
  int num_rows = 3;

  absl::Status result_status =
      sqlite_->SetTableContents(schema, contents, num_rows);
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(),
              HasSubstr("don't have the same number of columns"));
}

TEST_F(SetTableContentsTest, ZeroRows) {
  absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
      contents = CreateTableContents({}, {});

  CHECK_OK(sqlite_->SetTableContents(CreateInputTableSchema(), contents, 0));
}

class EvaluateQueryTest : public SetTableContentsTest {
 protected:
  void SetColumnNameAndType(ColumnSchema* col, std::string name,
                            sql_data::ColumnSchema_DataType type) {
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
                       sql_data::ColumnSchema_DataType_INT64);

  auto results = sqlite_->EvaluateQuery(query, output_schema);
  ASSERT_TRUE(results.ok());
  ASSERT_TRUE(results->contains(output_col_name));
  ASSERT_TRUE(results->at(output_col_name).has_int64_values());
  ASSERT_EQ(results->at(output_col_name).int64_values().value(0), 2);
}

TEST_F(EvaluateQueryTest, ValidQueryScalarFunction) {
  const std::string query = R"sql(
    SELECT ABS(-2) AS two
  )sql";
  std::string output_col_name = "two";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       sql_data::ColumnSchema_DataType_INT64);

  auto results = sqlite_->EvaluateQuery(query, output_schema);
  ASSERT_TRUE(results.ok());
  ASSERT_TRUE(results->contains(output_col_name));
  ASSERT_TRUE(results->at(output_col_name).has_int64_values());
  ASSERT_EQ(results->at(output_col_name).int64_values().value(0), 2);
}

TEST_F(EvaluateQueryTest, InvalidQueryParseError) {
  const std::string query = R"sql(
    BAD STATEMENT blah blah blah
  )sql";
  TableSchema empty_schema;

  auto result = sqlite_->EvaluateQuery(query, empty_schema);
  ASSERT_TRUE(absl::IsInvalidArgument(result.status()));
  ASSERT_THAT(result.status().message(), HasSubstr("syntax error"));
}

TEST_F(EvaluateQueryTest, InvalidQueryMissingFunction) {
  // `IF()` function is available in GoogleSQL but not SQLite.
  const std::string query = R"sql(
    SELECT IF(TRUE, 1, 0) AS one
  )sql";
  TableSchema empty_schema;

  auto result = sqlite_->EvaluateQuery(query, empty_schema);
  ASSERT_TRUE(absl::IsInvalidArgument(result.status()));
  ASSERT_THAT(result.status().message(), HasSubstr("no such function: IF"));
}

TEST_F(EvaluateQueryTest, InvalidQueryInvalidTable) {
  const std::string query = R"sql(
    SELECT val FROM missing_table
  )sql";
  TableSchema empty_schema;

  auto result = sqlite_->EvaluateQuery(query, empty_schema);
  ASSERT_TRUE(absl::IsInvalidArgument(result.status()));
  ASSERT_THAT(result.status().message(),
              HasSubstr("no such table: missing_table"));
}

TEST_F(EvaluateQueryTest, InvalidQueryInvalidColumn) {
  const std::string query = R"sql(
    SELECT missing_col FROM t
  )sql";
  TableSchema empty_schema;

  auto result = sqlite_->EvaluateQuery(query, empty_schema);
  ASSERT_TRUE(absl::IsInvalidArgument(result.status()));
  ASSERT_THAT(result.status().message(),
              HasSubstr("no such column: missing_col"));
}

TEST_F(EvaluateQueryTest, EmptyResults) {
  std::string output_col_name = "x";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       sql_data::ColumnSchema_DataType_INT64);

  auto results = sqlite_->EvaluateQuery(
      R"sql(
                            WITH D (x) AS (VALUES (1), (2), (3))
                            SELECT * FROM D WHERE FALSE;
                           )sql",
      output_schema);

  ASSERT_TRUE(results.ok());
  ASSERT_TRUE(results->contains(output_col_name));
  ASSERT_TRUE(!results->at(output_col_name).has_int64_values());
}

TEST_F(EvaluateQueryTest, ResultsFromTable) {
  absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
      contents = CreateTableContents({42}, {"a"});
  TableSchema schema = CreateInputTableSchema();
  int num_rows = 1;
  CHECK_OK(sqlite_->SetTableContents(schema, contents, num_rows));

  std::string output_col_name = "int_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       sql_data::ColumnSchema_DataType_INT64);

  auto results =
      sqlite_->EvaluateQuery(R"sql(SELECT int_vals FROM t;)sql", output_schema);
  ASSERT_TRUE(results.ok());
  ASSERT_TRUE(results->contains(output_col_name));
  ASSERT_TRUE(results->at(output_col_name).has_int64_values());
  ASSERT_EQ(results->at(output_col_name).int64_values().value(0), 42);
}

TEST_F(EvaluateQueryTest, SetTableContentsClearsPreviousContents) {
  absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
      contents = CreateTableContents({1, 2, 3}, {"a", "b", "c"});
  TableSchema schema = CreateInputTableSchema();
  int num_rows = 3;
  int kNumSetContents = 5;
  for (int i = 0; i < kNumSetContents; ++i) {
    CHECK_OK(sqlite_->SetTableContents(schema, contents, num_rows));
  }

  std::string output_col_name = "n";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       sql_data::ColumnSchema_DataType_INT64);

  auto results = sqlite_->EvaluateQuery(R"sql(SELECT COUNT(*) AS n FROM t;)sql",
                                        output_schema);
  ASSERT_TRUE(results.ok());
  ASSERT_TRUE(results->contains(output_col_name));
  ASSERT_TRUE(results->at(output_col_name).has_int64_values());
  ASSERT_EQ(results->at(output_col_name).int64_values().value(0), num_rows);
}

TEST_F(EvaluateQueryTest, MultipleResultRows) {
  absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
      contents = CreateTableContents({1, 2}, {"a", "b"});
  TableSchema schema = CreateInputTableSchema();
  int num_rows = 2;
  CHECK_OK(sqlite_->SetTableContents(schema, contents, num_rows));

  std::string output_col_name = "int_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       sql_data::ColumnSchema_DataType_INT64);

  auto results = sqlite_->EvaluateQuery(
      R"sql(SELECT int_vals FROM t ORDER BY int_vals;)sql", output_schema);

  ASSERT_TRUE(results.ok());
  ASSERT_TRUE(results->contains(output_col_name));
  ASSERT_TRUE(results->at(output_col_name).has_int64_values());
  ASSERT_EQ(results->at(output_col_name).int64_values().value(0), 1);
  ASSERT_EQ(results->at(output_col_name).int64_values().value(1), 2);
}

TEST_F(EvaluateQueryTest, MultipleColumns) {
  absl::flat_hash_map<std::string, ExampleQueryResult_VectorData_Values>
      contents = CreateTableContents({1}, {"a"});
  TableSchema schema = CreateInputTableSchema();
  int num_rows = 1;
  CHECK_OK(sqlite_->SetTableContents(schema, contents, num_rows));

  std::string int_output_col_name = "int_vals";
  std::string str_output_col_name = "str_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), int_output_col_name,
                       sql_data::ColumnSchema_DataType_INT64);
  SetColumnNameAndType(output_schema.add_column(), str_output_col_name,
                       sql_data::ColumnSchema_DataType_STRING);

  auto results = sqlite_->EvaluateQuery(
      R"sql(SELECT int_vals, str_vals FROM t;)sql", output_schema);

  ASSERT_TRUE(results.ok());
  ASSERT_TRUE(results->contains(int_output_col_name));
  ASSERT_TRUE(results->at(int_output_col_name).has_int64_values());
  ASSERT_EQ(results->at(int_output_col_name).int64_values().value(0), 1);
  ASSERT_TRUE(results->contains(str_output_col_name));
  ASSERT_TRUE(results->at(str_output_col_name).has_string_values());
  ASSERT_EQ(results->at(str_output_col_name).string_values().value(0), "a");
}

TEST_F(EvaluateQueryTest, IncorrectSchemaNumColumns) {
  std::string output_col_name = "v1";
  TableSchema output_schema;
  ColumnSchema* column = output_schema.add_column();
  column->set_name(output_col_name);
  column->set_type(sql_data::ColumnSchema_DataType_INT64);

  auto results = sqlite_->EvaluateQuery(
      R"sql(
                    WITH D (v1, v2) AS (VALUES (1, 2), (3, 4), (5, 6))
                    SELECT * FROM D;
                    )sql",
      output_schema);

  ASSERT_TRUE(absl::IsInvalidArgument(results.status()));
  ASSERT_THAT(
      results.status().message(),
      HasSubstr("Query results did not match the specified output schema"));
}

TEST_F(EvaluateQueryTest, IncorrectSchemaColumnNames) {
  std::string output_col_name = "v1";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       sql_data::ColumnSchema_DataType_INT64);

  auto results =
      sqlite_->EvaluateQuery(R"sql(SELECT 42 AS v)sql", output_schema);

  ASSERT_TRUE(absl::IsInvalidArgument(results.status()));
  ASSERT_THAT(
      results.status().message(),
      HasSubstr("Query results did not match the specified output schema"));
}

TEST_F(EvaluateQueryTest, AllSupportedDataTypes) {
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), "int32_val",
                       sql_data::ColumnSchema_DataType_INT32);
  SetColumnNameAndType(output_schema.add_column(), "int64_val",
                       sql_data::ColumnSchema_DataType_INT64);
  SetColumnNameAndType(output_schema.add_column(), "float_val",
                       sql_data::ColumnSchema_DataType_FLOAT);
  SetColumnNameAndType(output_schema.add_column(), "double_val",
                       sql_data::ColumnSchema_DataType_DOUBLE);
  SetColumnNameAndType(output_schema.add_column(), "bool_val",
                       sql_data::ColumnSchema_DataType_BOOL);
  SetColumnNameAndType(output_schema.add_column(), "str_val",
                       sql_data::ColumnSchema_DataType_STRING);
  SetColumnNameAndType(output_schema.add_column(), "binary_val",
                       sql_data::ColumnSchema_DataType_BYTES);

  auto results = sqlite_->EvaluateQuery(
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
      output_schema);

  LOG(INFO) << results.status().message();
  ASSERT_TRUE(results.ok());
}
