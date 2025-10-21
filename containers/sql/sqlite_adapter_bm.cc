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

// These benchmarks use the Benchmark library.  For instructions on how to run
// these benchmarks see:
// https://github.com/google/benchmark/blob/main/docs/user_guide.md#running-benchmarks

#include <thread>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "benchmark/benchmark.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/sql/row_set.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "fcp/protos/data_type.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::sql {

namespace {

using ::absl_testing::IsOk;
using ::confidential_federated_compute::fed_sql::CreateRowLocationsForAllRows;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::TableSchema;
using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::DynamicMessageFactory;
using ::google::protobuf::FieldDescriptorProto;
using ::google::protobuf::FileDescriptorProto;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;

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
        field {
          name: "str_vals"
          number: 2
          type: TYPE_STRING
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

  std::unique_ptr<Message> CreateMessage(int64_t val1,
                                         const std::string& val2) {
    std::unique_ptr<Message> message(prototype_->New());
    const Reflection* reflection = message->GetReflection();
    reflection->SetInt64(message.get(),
                         descriptor_->FindFieldByName("int_vals"), val1);
    reflection->SetString(message.get(),
                          descriptor_->FindFieldByName("str_vals"), val2);
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

class SqliteAdapterBenchmark : public benchmark::Fixture {
 public:
  std::unique_ptr<SqliteAdapter> sqlite_;
  void SetUp(::benchmark::State& state) override {
    CHECK_OK(SqliteAdapter::Initialize());
    absl::StatusOr<std::unique_ptr<SqliteAdapter>> create_status =
        SqliteAdapter::Create();
    CHECK_OK(create_status);
    sqlite_ = std::move(create_status.value());
  }

  void TearDown(::benchmark::State& state) override {
    sqlite_.reset();
    SqliteAdapter::ShutDown();
  }
};

TableSchema CreateInputTableSchema(
    absl::string_view table_name = "t",
    absl::string_view int_col_name = "int_vals",
    absl::string_view str_col_name = "str_vals") {
  TableSchema schema;
  schema.set_name(std::string(table_name));
  ColumnSchema* col1 = schema.add_column();
  col1->set_name(std::string(int_col_name));
  col1->set_type(google::internal::federated::plan::INT64);
  ColumnSchema* col2 = schema.add_column();
  col2->set_name(std::string(str_col_name));
  col2->set_type(google::internal::federated::plan::STRING);
  ColumnSchema* event_time_col = schema.add_column();
  event_time_col->set_name(kEventTimeColumnName);
  event_time_col->set_type(google::internal::federated::plan::STRING);
  const std::string create_table_stmt = absl::StrFormat(
      R"sql(CREATE TABLE %s (%s INTEGER, %s TEXT, %s TEXT))sql", table_name,
      int_col_name, str_col_name, kEventTimeColumnName);
  schema.set_create_table_sql(std::string(create_table_stmt));

  return schema;
}

absl::StatusOr<std::vector<Tensor>> CreateTableContents(
    const std::vector<int64_t>& int_vals,
    const std::vector<std::string>& str_vals,
    const std::vector<std::string>& event_times,
    absl::string_view int_col_name = "int_vals",
    absl::string_view str_col_name = "str_vals") {
  std::vector<Tensor> contents;

  FCP_ASSIGN_OR_RETURN(Tensor int_tensor,
                       Tensor::Create(DataType::DT_INT64,
                                      {static_cast<int64_t>(int_vals.size())},
                                      CreateTestData<int64_t>(int_vals),
                                      std::string(int_col_name)));

  FCP_ASSIGN_OR_RETURN(Tensor str_tensor,
                       Tensor::Create(DataType::DT_STRING,
                                      {static_cast<int64_t>(str_vals.size())},
                                      CreateStringTestData(str_vals),
                                      std::string(str_col_name)));

  FCP_ASSIGN_OR_RETURN(
      Tensor event_time_tensor,
      Tensor::Create(DataType::DT_STRING,
                     {static_cast<int64_t>(event_times.size())},
                     CreateStringTestData(event_times),
                     std::string(kEventTimeColumnName)));

  contents.push_back(std::move(int_tensor));
  contents.push_back(std::move(str_tensor));
  contents.push_back(std::move(event_time_tensor));
  return contents;
}

BENCHMARK_DEFINE_F(SqliteAdapterBenchmark, BM_AddTableContents)
(benchmark::State& state) {
  int num_rows = state.range(0);
  std::vector<int64_t> int_vals(num_rows);
  std::vector<std::string> str_vals(num_rows);
  std::vector<std::string> event_times(num_rows);

  for (int i = 0; i < num_rows; ++i) {
    int_vals[i] = i;
    str_vals[i] = absl::StrCat("long_not_inlined_row_name_", i);
    event_times[i] = absl::StrCat("2025-01-01 00:00:0", i);
  }
  absl::StatusOr<std::vector<Tensor>> contents =
      CreateTableContents(int_vals, str_vals, event_times);
  ASSERT_THAT(contents, IsOk());
  ASSERT_THAT(sqlite_->DefineTable(CreateInputTableSchema()), IsOk());

  std::vector<Input> storage;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents.value()), {});
  ASSERT_THAT(input, IsOk());
  storage.push_back(*std::move(input));
  std::vector<RowLocation> locations = CreateRowLocationsForAllRows(num_rows);
  absl::StatusOr<RowSet> row_set = RowSet::Create(locations, storage);
  ASSERT_THAT(row_set, IsOk());

  for (auto _ : state) {
    ASSERT_THAT(sqlite_->AddTableContents(*row_set), IsOk());
  }
}

BENCHMARK_REGISTER_F(SqliteAdapterBenchmark, BM_AddTableContents)
    ->Arg(1)
    ->Arg(5)
    ->Arg(10)
    ->Range(50, 1000000);

absl::StatusOr<std::vector<Input>> CreateMessageInput(
    std::vector<int64_t> int_vals, std::vector<std::string> str_vals,
    std::vector<std::string> event_times, MessageHelper& message_helper) {
  std::vector<std::unique_ptr<Message>> messages;
  for (int i = 0; i < int_vals.size(); ++i) {
    messages.push_back(message_helper.CreateMessage(int_vals[i], str_vals[i]));
  }

  std::vector<Tensor> system_columns;
  FCP_ASSIGN_OR_RETURN(
      Tensor event_time_tensor,
      Tensor::Create(DataType::DT_STRING,
                     {static_cast<int64_t>(event_times.size())},
                     CreateStringTestData(event_times), kEventTimeColumnName));
  system_columns.push_back(std::move(event_time_tensor));

  FCP_ASSIGN_OR_RETURN(
      Input input, Input::CreateFromMessages(std::move(messages),
                                             std::move(system_columns), {}));
  std::vector<Input> inputs;
  inputs.push_back(std::move(input));
  return inputs;
}

BENCHMARK_DEFINE_F(SqliteAdapterBenchmark, BM_MessageAddTableContents)
(benchmark::State& state) {
  int num_rows = state.range(0);
  std::vector<int64_t> int_vals(num_rows);
  std::vector<std::string> str_vals(num_rows);
  std::vector<std::string> event_times(num_rows);

  for (int i = 0; i < num_rows; ++i) {
    int_vals[i] = i;
    str_vals[i] = absl::StrCat("long_not_inlined_row_name_", i);
    event_times[i] = absl::StrCat("2025-01-01 00:00:0", i);
  }
  MessageHelper message_helper;
  absl::StatusOr<std::vector<Input>> inputs =
      CreateMessageInput(int_vals, str_vals, event_times, message_helper);
  ASSERT_THAT(inputs, IsOk());

  TableSchema schema = CreateInputTableSchema();
  ASSERT_THAT(sqlite_->DefineTable(schema), IsOk());

  std::vector<RowLocation> locations = CreateRowLocationsForAllRows(num_rows);
  absl::StatusOr<RowSet> row_set =
      RowSet::Create(locations, *std::move(inputs));
  ASSERT_THAT(row_set, IsOk());

  for (auto _ : state) {
    ASSERT_THAT(sqlite_->AddTableContents(*row_set), IsOk());
  }
}

BENCHMARK_REGISTER_F(SqliteAdapterBenchmark, BM_MessageAddTableContents)
    ->Arg(1)
    ->Arg(5)
    ->Arg(10)
    ->Range(50, 1000000);

}  // namespace
}  // namespace confidential_federated_compute::sql

BENCHMARK_MAIN();
