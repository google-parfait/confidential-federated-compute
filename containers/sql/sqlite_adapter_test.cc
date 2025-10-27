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

#include <thread>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "containers/sql/input.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "fcp/protos/data_type.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::sql {

namespace {

using ::absl_testing::IsOk;
using ::fcp::client::ExampleQueryResult_VectorData;
using ::fcp::client::ExampleQueryResult_VectorData_Int64Values;
using ::fcp::client::ExampleQueryResult_VectorData_StringValues;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::TableSchema;
using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::DynamicMessageFactory;
using ::google::protobuf::FieldDescriptorProto;
using ::google::protobuf::FileDescriptorProto;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;
using ::tensorflow_federated::aggregation::AggVector;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorData;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::ContainerEq;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Test;

class MessageHelper {
 public:
  MessageHelper() {
    const FileDescriptorProto file_proto = PARSE_TEXT_PROTO(R"pb(
      name: "test.proto"
      package: "confidential_federated_compute.sql"
      message_type {
        name: "TestMessage"
        field {
          name: "int_vals"
          number: 1
          type: TYPE_INT64
          label: LABEL_OPTIONAL
        }
      }
    )pb");
    const google::protobuf::FileDescriptor* file_descriptor =
        pool_.BuildFile(file_proto);
    CHECK_NE(file_descriptor, nullptr);

    descriptor_ = file_descriptor->FindMessageTypeByName("TestMessage");
    CHECK_NE(descriptor_, nullptr);

    prototype_ = factory_.GetPrototype(descriptor_);
  }

  std::unique_ptr<Message> CreateMessage(int64_t val1) {
    std::unique_ptr<Message> message(prototype_->New());
    const Reflection* reflection = message->GetReflection();
    reflection->SetInt64(message.get(),
                         descriptor_->FindFieldByName("int_vals"), val1);
    return message;
  }

 private:
  DescriptorPool pool_;
  DynamicMessageFactory factory_{&pool_};
  const Descriptor* descriptor_;
  const Message* prototype_;
};

// Creates test tensor data based on a vector<T>.
template <typename T>
std::unique_ptr<MutableVectorData<T>> CreateTestData(
    const std::vector<T>& values) {
  return std::make_unique<MutableVectorData<T>>(values.begin(), values.end());
}

std::unique_ptr<MutableStringData> CreateStringTestData(
    const std::vector<std::string>& data) {
  std::unique_ptr<MutableStringData> tensor_data =
      std::make_unique<MutableStringData>(data.size());
  for (std::string value : data) {
    tensor_data->Add(std::move(value));
  }
  return tensor_data;
}

void SetColumnNameAndType(ColumnSchema* col, std::string name,
                          google::internal::federated::plan::DataType type) {
  col->set_name(name);
  col->set_type(type);
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
  col1->set_type(google::internal::federated::plan::INT64);
  ColumnSchema* col2 = schema.add_column();
  col2->set_name(std::string(col2_name));
  col2->set_type(google::internal::federated::plan::STRING);
  const std::string create_table_stmt =
      absl::StrFormat(R"sql(CREATE TABLE %s (%s INTEGER, %s TEXT))sql",
                      table_name, col1_name, col2_name);
  schema.set_create_table_sql(std::string(create_table_stmt));

  return schema;
}

TEST(MultipleThreadsTest, ConcurrentDefineTable) {
  ASSERT_THAT(SqliteAdapter::Initialize(), IsOk());
  std::thread thread1([]() {
    absl::StatusOr<std::unique_ptr<SqliteAdapter>> create_status =
        SqliteAdapter::Create();
    ASSERT_THAT(create_status, IsOk());
    std::unique_ptr<SqliteAdapter> adapter = std::move(create_status.value());
    ASSERT_THAT(adapter->DefineTable(CreateInputTableSchema("table_1")),
                IsOk());
  });
  std::thread thread2([]() {
    absl::StatusOr<std::unique_ptr<SqliteAdapter>> create_status =
        SqliteAdapter::Create();
    ASSERT_THAT(create_status, IsOk());
    std::unique_ptr<SqliteAdapter> adapter = std::move(create_status.value());
    ASSERT_THAT(adapter->DefineTable(CreateInputTableSchema("table_2")),
                IsOk());
  });
  thread1.join();
  thread2.join();
  SqliteAdapter::ShutDown();
}

class DefineTableTest : public SqliteAdapterTest {};

TEST_F(DefineTableTest, ValidCreateTableStatement) {
  ASSERT_THAT(sqlite_->DefineTable(CreateInputTableSchema()), IsOk());
}

TEST_F(DefineTableTest, InvalidCreateTableStatement) {
  TableSchema schema;
  schema.set_create_table_sql("BAD STATEMENT blah blah");
  absl::Status result_status = sqlite_->DefineTable(schema);
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(), HasSubstr("syntax error"));
}

class AddTableContentsTest : public SqliteAdapterTest {
 protected:
  std::vector<RowLocation> CreateRowLocations(int num_rows) {
    std::vector<RowLocation> locations;
    locations.reserve(num_rows);
    for (int i = 0; i < num_rows; ++i) {
      locations.push_back(
          {.dp_unit_hash = 0, .input_index = 0, .row_index = (uint32_t)i});
    }
    return locations;
  }

  absl::StatusOr<std::vector<Tensor>> CreateTableContents(
      const std::vector<int64_t>& int_vals,
      const std::vector<std::string>& str_vals,
      std::string int_col_name = "int_vals",
      std::string str_col_name = "str_vals") {
    std::vector<Tensor> contents;

    FCP_ASSIGN_OR_RETURN(
        Tensor int_tensor,
        Tensor::Create(DataType::DT_INT64,
                       {static_cast<int64_t>(int_vals.size())},
                       CreateTestData<int64_t>(int_vals), int_col_name));

    FCP_ASSIGN_OR_RETURN(
        Tensor str_tensor,
        Tensor::Create(DataType::DT_STRING,
                       {static_cast<int64_t>(str_vals.size())},
                       CreateStringTestData(str_vals), str_col_name));

    contents.push_back(std::move(int_tensor));
    contents.push_back(std::move(str_tensor));
    return contents;
  }
};

TEST_F(AddTableContentsTest, TensorBasicUsage) {
  ASSERT_THAT(sqlite_->DefineTable(CreateInputTableSchema()), IsOk());
  absl::StatusOr<std::vector<Tensor>> contents =
      CreateTableContents({1, 2, 3}, {"a", "b", "c"});
  ASSERT_THAT(contents, IsOk());

  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents.value()), {});
  ASSERT_THAT(input, IsOk());
  std::vector<Input> storage;
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(3);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());
  ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());
}

TEST_F(AddTableContentsTest, MessageDataBasicUsage) {
  ASSERT_THAT(sqlite_->DefineTable(CreateInputTableSchema()), IsOk());
  std::vector<std::unique_ptr<Message>> messages;
  MessageHelper message_helper;
  messages.push_back(message_helper.CreateMessage(1));
  messages.push_back(message_helper.CreateMessage(2));

  std::vector<Tensor> system_columns;
  absl::StatusOr<Tensor> event_time_tensor =
      Tensor::Create(DataType::DT_STRING, TensorShape({2}),
                     CreateTestData<absl::string_view>({"a", "b"}), "str_vals");
  ASSERT_THAT(event_time_tensor, IsOk());
  system_columns.push_back(*std::move(event_time_tensor));

  std::vector<Input> storage;
  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages), std::move(system_columns), {});
  ASSERT_THAT(input, IsOk());
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(2);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());
  ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());
}

TEST_F(AddTableContentsTest, ColumnNameEscaping) {
  TableSchema schema = CreateInputTableSchema(
      /*table_name=*/"t", /*col1_name=*/"![]^'\";/int_vals",
      /*col2_name=*/"![]^'\";/str_vals");
  // CreateInputTableSchema doesn't properly escape create_table_sql.
  schema.set_create_table_sql(
      "CREATE TABLE t ('![]^''\";/int_vals' INTEGER, '![]^''\";/str_vals' "
      "TEXT)");

  ASSERT_THAT(sqlite_->DefineTable(schema), IsOk());
  absl::StatusOr<std::vector<Tensor>> contents =
      CreateTableContents({1, 2, 3}, {"a", "b", "c"}, schema.column(0).name(),
                          schema.column(1).name());
  ASSERT_THAT(contents, IsOk());

  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents.value()), {});
  ASSERT_THAT(input, IsOk());
  std::vector<Input> storage;
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(3);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());
  ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());
}

TEST_F(AddTableContentsTest, CalledBeforeDefineTable) {
  absl::StatusOr<std::vector<Tensor>> contents =
      CreateTableContents({1}, {"a"});
  ASSERT_THAT(contents, IsOk());

  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents.value()), {});
  ASSERT_THAT(input, IsOk());
  std::vector<Input> storage;
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(1);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());

  absl::Status result_status = sqlite_->AddTableContents(*std::move(row_set));
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(),
              HasSubstr("`DefineTable` must be called before"));
}

TEST_F(AddTableContentsTest, ZeroRows) {
  ASSERT_THAT(sqlite_->DefineTable(CreateInputTableSchema()), IsOk());
  absl::StatusOr<std::vector<Tensor>> contents = CreateTableContents({}, {});
  TableSchema schema = CreateInputTableSchema();
  ASSERT_THAT(contents, IsOk());

  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents.value()), {});
  ASSERT_THAT(input, IsOk());
  std::vector<Input> storage;
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(0);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());
  ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());
}

// Converts a potentially sparse tensor to a flat vector of tensor values.
template <typename T>
std::vector<T> TensorValuesToVector(const Tensor& arg) {
  std::vector<T> vec(arg.num_elements());
  if (arg.num_elements() > 0) {
    AggVector<T> agg_vector = arg.AsAggVector<T>();
    for (auto [i, v] : agg_vector) {
      vec[i] = v;
    }
  }
  return vec;
}

TEST_F(AddTableContentsTest, BatchedInserts) {
  ASSERT_THAT(sqlite_->DefineTable(CreateInputTableSchema()), IsOk());
  // Add enough rows to trigger batching.
  int num_rows = SqliteAdapter::kSqliteVariableLimit * 2 + 1;
  std::vector<int64_t> int_vals(num_rows);
  std::vector<std::string> str_vals(num_rows);

  for (int i = 0; i < num_rows; ++i) {
    int_vals[i] = i;
    str_vals[i] = absl::StrCat("row_", i);
  }

  absl::StatusOr<std::vector<Tensor>> contents =
      CreateTableContents(int_vals, str_vals);
  ASSERT_THAT(contents, IsOk());

  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents.value()), {});
  ASSERT_THAT(input, IsOk());
  std::vector<Input> storage;
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(num_rows);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());
  ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());

  // Verify the data was inserted correctly
  std::string output_col_name = "int_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       google::internal::federated::plan::INT64);

  auto result_status = sqlite_->EvaluateQuery(
      "SELECT int_vals FROM t ORDER BY int_vals;", output_schema.column());

  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  EXPECT_EQ(result.size(), 1);
  EXPECT_THAT(result[0].num_elements(), Eq(num_rows));
  EXPECT_THAT(TensorValuesToVector<int64_t>(result[0]), ContainerEq(int_vals));
}

TEST_F(AddTableContentsTest, RowSetWithExtraSystemColumns) {
  ASSERT_THAT(sqlite_->DefineTable(CreateInputTableSchema()), IsOk());
  // Create a RowSet with an extra system column that isn't present in the table
  // schema.
  std::vector<std::unique_ptr<Message>> messages;
  MessageHelper message_helper;
  messages.push_back(message_helper.CreateMessage(1));

  std::vector<Tensor> system_columns;

  absl::StatusOr<Tensor> extra_tensor = Tensor::Create(
      DataType::DT_INT64, {1}, CreateTestData<int64_t>({99}), "extra_col");
  ASSERT_THAT(extra_tensor, IsOk());
  system_columns.push_back(*std::move(extra_tensor));

  absl::StatusOr<Tensor> str_tensor =
      Tensor::Create(DataType::DT_STRING, {1},
                     CreateTestData<absl::string_view>({"a"}), "str_vals");
  ASSERT_THAT(str_tensor, IsOk());
  system_columns.push_back(*std::move(str_tensor));

  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages), std::move(system_columns), {});
  ASSERT_THAT(input, IsOk());
  std::vector<Input> storage;
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(1);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());
  ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());

  // Verify the data was inserted correctly and the extra column was ignored.
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), "int_vals",
                       google::internal::federated::plan::INT64);
  SetColumnNameAndType(output_schema.add_column(), "str_vals",
                       google::internal::federated::plan::STRING);

  auto result_status =
      sqlite_->EvaluateQuery("SELECT * FROM t;", output_schema.column());

  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  // Verify that the extra column was not inserted.
  ASSERT_EQ(result.size(), 2);
}

TEST_F(AddTableContentsTest, RowSetMissingTableSchemaColumn) {
  ASSERT_THAT(sqlite_->DefineTable(CreateInputTableSchema()), IsOk());

  std::vector<Tensor> contents;
  absl::StatusOr<Tensor> int_tensor = Tensor::Create(
      DataType::DT_INT64, {1}, CreateTestData<int64_t>({1}), "int_vals");
  ASSERT_THAT(int_tensor, IsOk());
  contents.push_back(*std::move(int_tensor));

  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents), {});
  ASSERT_THAT(input, IsOk());
  std::vector<Input> storage;
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(1);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());

  absl::Status result_status = sqlite_->AddTableContents(*std::move(row_set));
  ASSERT_TRUE(absl::IsInvalidArgument(result_status));
  ASSERT_THAT(result_status.message(),
              HasSubstr("Row does not contain a column present in the table "
                        "schema: str_vals"));
}

class EvaluateQueryTest : public AddTableContentsTest {
 protected:
  void SetUp() override {
    AddTableContentsTest::SetUp();
    ASSERT_THAT(sqlite_->DefineTable(CreateInputTableSchema()), IsOk());
  }
};

TEST_F(EvaluateQueryTest, ValidQueryBasicExpression) {
  const std::string query = R"sql(
    SELECT 1+1 AS two
  )sql";
  std::string output_col_name = "two";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       google::internal::federated::plan::INT64);

  auto result_status = sqlite_->EvaluateQuery(query, output_schema.column());
  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).num_elements(), 1);
  ASSERT_EQ(result.at(0).AsSpan<int64_t>().at(0), 2);
}

TEST_F(EvaluateQueryTest, ValidQueryBasicStringExpression) {
  const std::string query = R"sql(
    SELECT 'foo' AS str_val
  )sql";
  std::string output_col_name = "str_val";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       google::internal::federated::plan::STRING);

  auto result_status = sqlite_->EvaluateQuery(query, output_schema.column());
  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).dtype(), DataType::DT_STRING);
  ASSERT_EQ(result.at(0).num_elements(), 1);
  ASSERT_EQ(result.at(0).AsSpan<absl::string_view>().at(0), "foo");
}

TEST_F(EvaluateQueryTest, ValidQueryScalarFunction) {
  const std::string query = R"sql(
    SELECT ABS(-2) AS two
  )sql";
  std::string output_col_name = "two";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       google::internal::federated::plan::INT64);

  auto result_status = sqlite_->EvaluateQuery(query, output_schema.column());
  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).num_elements(), 1);
  ASSERT_EQ(result.at(0).AsSpan<int64_t>().at(0), 2);
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
                       google::internal::federated::plan::INT64);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(
                            WITH D (x) AS (VALUES (1), (2), (3))
                            SELECT * FROM D WHERE FALSE;
                           )sql",
      output_schema.column());

  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).num_elements(), 0);
}

TEST_F(EvaluateQueryTest, TensorContentsResultsFromTable) {
  absl::StatusOr<std::vector<Tensor>> contents =
      CreateTableContents({42}, {"a"});
  ASSERT_THAT(contents, IsOk());

  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents.value()), {});
  ASSERT_THAT(input, IsOk());
  std::vector<Input> storage;
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(1);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());
  ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());

  std::string output_col_name = "int_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       google::internal::federated::plan::INT64);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(SELECT int_vals FROM t;)sql", output_schema.column());
  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).num_elements(), 1);
  ASSERT_EQ(result.at(0).AsSpan<int64_t>().at(0), 42);
}

TEST_F(EvaluateQueryTest, MessageContentsResultsFromTable) {
  std::vector<std::unique_ptr<Message>> messages;
  MessageHelper message_helper;
  messages.push_back(message_helper.CreateMessage(42));
  messages.push_back(message_helper.CreateMessage(24));

  std::vector<Tensor> system_columns;
  absl::StatusOr<Tensor> str_vals_tensor = Tensor::Create(
      DataType::DT_STRING, TensorShape({2}),
      CreateTestData<absl::string_view>({"foo", "bar"}), "str_vals");
  ASSERT_THAT(str_vals_tensor, IsOk());
  system_columns.push_back(*std::move(str_vals_tensor));

  std::vector<Input> storage;
  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages), std::move(system_columns), {});
  ASSERT_THAT(input, IsOk());
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(2);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());
  ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());

  std::string output_col_name = "int_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       google::internal::federated::plan::INT64);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(SELECT int_vals FROM t ORDER BY int_vals;)sql",
      output_schema.column());
  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).num_elements(), 2);
  ASSERT_EQ(result.at(0).AsSpan<int64_t>().at(0), 24);
  ASSERT_EQ(result.at(0).AsSpan<int64_t>().at(1), 42);
}

TEST_F(EvaluateQueryTest, MultipleAddTableContents) {
  int num_rows = 3;
  int kNumSetContents = 5;
  for (int i = 0; i < kNumSetContents; ++i) {
    absl::StatusOr<std::vector<Tensor>> contents =
        CreateTableContents({1, 2, 3}, {"a", "b", "c"});
    ASSERT_THAT(contents, IsOk());

    absl::StatusOr<Input> input =
        Input::CreateFromTensors(std::move(contents.value()), {});
    ASSERT_THAT(input, IsOk());
    std::vector<Input> storage;
    storage.push_back(*std::move(input));
    std::vector<RowLocation> locations = CreateRowLocations(num_rows);
    absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
    ASSERT_THAT(row_set, IsOk());
    ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());
  }

  std::string output_col_name = "n";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       google::internal::federated::plan::INT64);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(SELECT COUNT(*) AS n FROM t;)sql", output_schema.column());
  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).num_elements(), 1);
  ASSERT_EQ(result.at(0).AsSpan<int64_t>().at(0), num_rows * kNumSetContents);
}

TEST_F(EvaluateQueryTest, MultipleResultRows) {
  absl::StatusOr<std::vector<Tensor>> contents =
      CreateTableContents({3, 2}, {"a", "b"});
  ASSERT_THAT(contents, IsOk());
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents.value()), {});
  ASSERT_THAT(input, IsOk());
  std::vector<Input> storage;
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(2);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());
  ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());

  std::string output_col_name = "int_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), output_col_name,
                       google::internal::federated::plan::INT64);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(SELECT int_vals FROM t ORDER BY int_vals;)sql",
      output_schema.column());

  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result.at(0).dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).num_elements(), 2);
  ASSERT_EQ(result.at(0).AsSpan<int64_t>().at(0), 2);
  ASSERT_EQ(result.at(0).AsSpan<int64_t>().at(1), 3);
}

TEST_F(EvaluateQueryTest, MultipleColumns) {
  absl::StatusOr<std::vector<Tensor>> contents =
      CreateTableContents({1}, {"a"});
  ASSERT_THAT(contents, IsOk());

  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents.value()), {});
  ASSERT_THAT(input, IsOk());
  std::vector<Input> storage;
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocations(1);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());
  ASSERT_THAT(sqlite_->AddTableContents(*std::move(row_set)), IsOk());

  std::string int_output_col_name = "int_vals";
  std::string str_output_col_name = "str_vals";
  TableSchema output_schema;
  SetColumnNameAndType(output_schema.add_column(), int_output_col_name,
                       google::internal::federated::plan::INT64);
  SetColumnNameAndType(output_schema.add_column(), str_output_col_name,
                       google::internal::federated::plan::STRING);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(SELECT int_vals, str_vals FROM t;)sql", output_schema.column());

  ASSERT_THAT(result_status, IsOk());
  std::vector<Tensor> result = std::move(result_status.value());
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result.at(0).dtype(), DataType::DT_INT64);
  ASSERT_EQ(result.at(0).num_elements(), 1);
  ASSERT_EQ(result.at(0).AsSpan<int64_t>().at(0), 1);
  ASSERT_EQ(result.at(1).dtype(), DataType::DT_STRING);
  ASSERT_EQ(result.at(1).num_elements(), 1);
  ASSERT_EQ(result.at(1).AsSpan<absl::string_view>().at(0), "a");
}

TEST_F(EvaluateQueryTest, IncorrectSchemaNumColumns) {
  std::string output_col_name = "v1";
  TableSchema output_schema;
  ColumnSchema* column = output_schema.add_column();
  column->set_name(output_col_name);
  column->set_type(google::internal::federated::plan::INT64);

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
                       google::internal::federated::plan::INT64);

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
                       google::internal::federated::plan::INT32);
  SetColumnNameAndType(output_schema.add_column(), "int64_val",
                       google::internal::federated::plan::INT64);
  SetColumnNameAndType(output_schema.add_column(), "float_val",
                       google::internal::federated::plan::FLOAT);
  SetColumnNameAndType(output_schema.add_column(), "double_val",
                       google::internal::federated::plan::DOUBLE);
  SetColumnNameAndType(output_schema.add_column(), "bool_val",
                       google::internal::federated::plan::BOOL);
  SetColumnNameAndType(output_schema.add_column(), "str_val",
                       google::internal::federated::plan::STRING);
  SetColumnNameAndType(output_schema.add_column(), "binary_val",
                       google::internal::federated::plan::BYTES);

  auto result_status = sqlite_->EvaluateQuery(
      R"sql(
                            SELECT
                              1 AS int32_val,
                              1 AS int64_val,
                              1.0 AS float_val,
                              1.0 AS double_val,
                              TRUE AS bool_val,
                              'foobar' AS str_val,
                              'arbitrary bytes' AS binary_val;
                            )sql",
      output_schema.column());

  ASSERT_THAT(result_status, IsOk());
}

}  // namespace

}  // namespace confidential_federated_compute::sql
