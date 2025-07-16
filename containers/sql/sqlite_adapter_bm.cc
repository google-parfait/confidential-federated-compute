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
#include "absl/strings/str_format.h"
#include "benchmark/benchmark.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sql {

namespace {

using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::TableSchema;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_INT64;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_STRING;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;

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
  col1->set_type(ExampleQuerySpec_OutputVectorSpec_DataType_INT64);
  ColumnSchema* col2 = schema.add_column();
  col2->set_name(std::string(str_col_name));
  col2->set_type(ExampleQuerySpec_OutputVectorSpec_DataType_STRING);
  const std::string create_table_stmt =
      absl::StrFormat(R"sql(CREATE TABLE %s (%s INTEGER, %s TEXT))sql",
                      table_name, int_col_name, str_col_name);
  schema.set_create_table_sql(std::string(create_table_stmt));

  return schema;
}

absl::StatusOr<std::vector<Tensor>> CreateTableContents(
    const std::vector<int64_t>& int_vals,
    const std::vector<std::string>& str_vals,
    absl::string_view int_col_name = "int_vals",
    absl::string_view str_col_name = "str_vals") {
  std::vector<Tensor> contents;

  FCP_ASSIGN_OR_RETURN(Tensor int_tensor,
                       Tensor::Create(DataType::DT_INT64,
                                      {static_cast<int64_t>(int_vals.size())},
                                      CreateTestData<int64_t>(int_vals)));

  FCP_ASSIGN_OR_RETURN(Tensor str_tensor,
                       Tensor::Create(DataType::DT_STRING,
                                      {static_cast<int64_t>(str_vals.size())},
                                      CreateStringTestData(str_vals)));

  contents.push_back(std::move(int_tensor));
  contents.push_back(std::move(str_tensor));
  return contents;
}

BENCHMARK_DEFINE_F(SqliteAdapterBenchmark, BM_AddTableContents)
(benchmark::State& state) {
  int num_rows = state.range(0);
  std::vector<int64_t> int_vals(num_rows);
  std::vector<std::string> str_vals(num_rows);

  for (int i = 0; i < num_rows; ++i) {
    int_vals[i] = i;
    str_vals[i] = absl::StrCat("row_", i);
  }
  absl::StatusOr<std::vector<Tensor>> contents =
      CreateTableContents(int_vals, str_vals);
  CHECK_OK(contents);
  CHECK_OK(sqlite_->DefineTable(CreateInputTableSchema()));

  for (auto _ : state) {
    CHECK_OK(sqlite_->AddTableContents(contents.value(), num_rows));
  }
}

BENCHMARK_REGISTER_F(SqliteAdapterBenchmark, BM_AddTableContents)
    ->Arg(1)
    ->Arg(5)
    ->Arg(10)
    ->Arg(20)
    ->Arg(40)
    ->Range(50, 10000);

}  // namespace
}  // namespace confidential_federated_compute::sql

BENCHMARK_MAIN();
