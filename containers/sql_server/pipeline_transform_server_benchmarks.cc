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
// These benchmarks use the Benchmark library. For instructions on how to run
// these benchmarks see:
// https://github.com/google/benchmark/blob/main/docs/user_guide.md#running-benchmarks
#include "absl/log/check.h"
#include "benchmark/benchmark.h"
#include "containers/sql_server/pipeline_transform_server.h"
#include "containers/sql_server/sql_data.pb.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::sql_server {

namespace {

using ::fcp::aggregation::CheckpointBuilder;
using ::fcp::aggregation::CreateTestData;
using ::fcp::aggregation::DataType;
using ::fcp::aggregation::FederatedComputeCheckpointBuilderFactory;
using ::fcp::aggregation::FederatedComputeCheckpointParserFactory;
using ::fcp::aggregation::MutableVectorData;
using ::fcp::aggregation::Tensor;
using ::fcp::aggregation::TensorShape;
using ::fcp::client::ExampleQueryResult_VectorData;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::StatusCode;
using ::sql_data::ColumnSchema;
using ::sql_data::DatabaseSchema;
using ::sql_data::SqlData;
using ::sql_data::SqlQuery;
using ::sql_data::TableSchema;
using ::testing::HasSubstr;
using ::testing::Test;

inline constexpr absl::string_view kComplexQuery = R"(
  WITH Clause1 AS (
    SELECT
      column_1 AS a
    FROM t
    WHERE
      column_2 = 1
  ),
  Clause2 AS (
    SELECT
      CAST(column_3 AS INT64) as b,
      column_4,
      column_5,
      column_6 AS c,
      column_7 AS a,
    FROM t
  ),
  Clause3 AS (
    SELECT
      a,
      b,
      c,
      CAST(column_4 / 100 AS INT64) AS d,
      SUM(b) AS e,
      COUNT(*) AS f,
      SUM(column_4 - column_5) AS g
    FROM Clause2
    GROUP BY a, b, c, d
  ),
  Clause4 AS (
    SELECT
      Clause3.a,
      Clause3.b,
      Clause3.c,
      Clause3.d
      Clause3.e
      Clause3.f
      Clause3.g
    FROM Clause1
    INNER JOIN Clause3
      USING (a)
  )
  SELECT
    a,
    b,
    c,
    d,
    e,
    f,
    g,
    IFNULL(g, 0) AS h
  FROM Clause3
  LEFT JOIN Clause4
    USING (b, c, d, e)
  )";

TableSchema CreateTableSchema(std::string name, std::string create_table_sql) {
  TableSchema schema;
  schema.set_name(name);
  schema.set_create_table_sql(create_table_sql);
  return schema;
}

SqlQuery CreateSqlQuery(TableSchema input_table_schema,
                        absl::string_view raw_query,
                        TableSchema output_table_schema) {
  SqlQuery query;
  DatabaseSchema* input_schema = query.mutable_input_schema();
  DatabaseSchema* output_schema = query.mutable_output_schema();
  *(input_schema->add_table()) = input_table_schema;
  *(output_schema->add_table()) = output_table_schema;
  query.set_raw_sql(std::string(raw_query));
  return query;
}

ColumnSchema CreateColumnSchema(std::string name, ColumnSchema::DataType type) {
  ColumnSchema schema;
  schema.set_name(name);
  schema.set_type(type);
  return schema;
}

TableSchema CreateGenericTableSchema() {
  TableSchema schema = CreateTableSchema(
      "t",
      "CREATE TABLE t (column_1 INTEGER, column_2 INTEGER, column_3 "
      "INTEGER, column_4 INTEGER, column_5 INTEGER, column_6 INTEGER, column_7 "
      "INTEGER, column_8 INTEGER, column_9 INTEGER, column_10 INTEGER, "
      "column_11 INTEGER)");

  *(schema.add_column()) = CreateColumnSchema("column_1", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("column_2", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("column_3", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("column_4", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("column_5", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("column_6", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("column_7", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("column_8", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("column_9", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("column_10", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("column_11", ColumnSchema::INT64);

  return schema;
}

TableSchema CreateComplexQueryOutputTableSchema() {
  TableSchema schema = CreateTableSchema("output",
                                         "CREATE TABLE output("
                                         "a INT64,"
                                         "b INT64,"
                                         "c INT64,"
                                         "d INT64,"
                                         "e INT64,"
                                         "f INT64,"
                                         "g INT64,"
                                         "h INT64)");

  *(schema.add_column()) = CreateColumnSchema("a", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("b", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("c", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("d", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("e", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("f", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("g", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("h", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("i", ColumnSchema::INT64);
  *(schema.add_column()) = CreateColumnSchema("j", ColumnSchema::INT64);

  return schema;
}

std::unique_ptr<MutableVectorData<uint64_t>> GenerateInt64InputColumn(
    int num_rows) {
  std::vector<uint64_t> input_column;
  input_column.reserve(num_rows);
  for (int i = 0; i < num_rows; i++) {
    input_column.push_back(std::rand());
  }
  return std::make_unique<MutableVectorData<uint64_t>>(input_column.begin(),
                                                       input_column.end());
}

std::string BuildRandomInt64SingleClientData(int num_rows) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  for (int i = 1; i <= 11; i++) {
    std::string column_name = absl::StrFormat("column_%d", i);
    absl::StatusOr<Tensor> t1 =
        Tensor::Create(DataType::DT_INT64, TensorShape({num_rows}),
                       GenerateInt64InputColumn(num_rows));
    CHECK_OK(t1);
    CHECK_OK(ckpt_builder->Add(column_name, *t1));
  }

  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

class PipelineTransformBenchmark : public benchmark::Fixture {
 public:
  void SetUp(::benchmark::State& state) override {
    int port;
    service_ = std::make_unique<SqlPipelineTransform>();
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    channel_ = grpc::CreateChannel(server_address + std::to_string(port),
                                   grpc::InsecureChannelCredentials());
    stub_ = PipelineTransform::NewStub(channel_);
  }

  void TearDown(::benchmark::State& state) override {
    stub_.reset(nullptr);
    channel_.reset();
    server_->Shutdown();
    server_.reset(nullptr);
    service_.reset(nullptr);
  }

 protected:
  std::unique_ptr<SqlPipelineTransform> service_ = nullptr;
  std::unique_ptr<Server> server_ = nullptr;
  std::shared_ptr<grpc::Channel> channel_ = nullptr;
  std::unique_ptr<PipelineTransform::Stub> stub_ = nullptr;
};

BENCHMARK_DEFINE_F(PipelineTransformBenchmark, BM_TransformInputAgnosticQuery)
(benchmark::State& state) {
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  TableSchema input_schema = CreateGenericTableSchema();
  TableSchema output_schema =
      CreateTableSchema("output", "CREATE TABLE output (one INTEGER)");
  *(output_schema.add_column()) =
      CreateColumnSchema("one", ColumnSchema::INT64);

  SqlQuery query =
      CreateSqlQuery(input_schema, "SELECT 1 AS one", output_schema);
  configure_request.mutable_configuration()->PackFrom(query);
  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildRandomInt64SingleClientData(state.range(0)));
  TransformResponse transform_response;
  for (auto _ : state) {
    grpc::ClientContext transform_context;
    auto transform_status = stub_->Transform(
        &transform_context, transform_request, &transform_response);
  }
}

BENCHMARK_REGISTER_F(PipelineTransformBenchmark, BM_TransformInputAgnosticQuery)
    ->Range(1, 5000);

BENCHMARK_DEFINE_F(PipelineTransformBenchmark, BM_TransformSumAllColumns)
(benchmark::State& state) {
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  TableSchema input_schema = CreateGenericTableSchema();
  TableSchema output_schema = CreateGenericTableSchema();

  SqlQuery query =
      CreateSqlQuery(input_schema,
                     "SELECT SUM(column_1) AS column_1, SUM(column_2) AS "
                     "column_2, SUM(column_3) AS column_3, SUM(column_4) AS "
                     "column_4, SUM(column_5) AS column_5, SUM(column_6) AS "
                     "column_6, SUM(column_7) AS column_7, SUM(column_8) AS "
                     "column_8, SUM(column_9) AS column_9, SUM(column_10) AS"
                     "column_10, SUM(column_11) AS column_11 FROM t",
                     output_schema);
  configure_request.mutable_configuration()->PackFrom(query);

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildRandomInt64SingleClientData(state.range(0)));
  TransformResponse transform_response;

  for (auto _ : state) {
    grpc::ClientContext transform_context;
    auto transform_status = stub_->Transform(
        &transform_context, transform_request, &transform_response);
  }
}

BENCHMARK_REGISTER_F(PipelineTransformBenchmark, BM_TransformSumAllColumns)
    ->Range(1, 5000);

BENCHMARK_DEFINE_F(PipelineTransformBenchmark, BM_TransformSumGroupBy)
(benchmark::State& state) {
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  TableSchema input_schema = CreateGenericTableSchema();
  TableSchema output_schema = CreateGenericTableSchema();

  SqlQuery query = CreateSqlQuery(
      input_schema,
      "SELECT column_1, SUM(column_2) AS "
      "column_2, SUM(column_3) AS column_3, SUM(column_4) AS "
      "column_4, SUM(column_5) AS column_5, SUM(column_6) AS "
      "column_6, SUM(column_7) AS column_7, SUM(column_8) AS "
      "column_8, SUM(column_9) AS column_9, SUM(column_10) AS "
      "column_10, SUM(column_11) AS column_11 FROM t GROUP BY column_1",
      output_schema);
  configure_request.mutable_configuration()->PackFrom(query);

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildRandomInt64SingleClientData(state.range(0)));
  TransformResponse transform_response;

  for (auto _ : state) {
    grpc::ClientContext transform_context;
    auto transform_status = stub_->Transform(
        &transform_context, transform_request, &transform_response);
  }
}

BENCHMARK_REGISTER_F(PipelineTransformBenchmark, BM_TransformSumGroupBy)
    ->Range(1, 5000);

BENCHMARK_DEFINE_F(PipelineTransformBenchmark, BM_TransformSelectAll)
(benchmark::State& state) {
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  TableSchema input_schema = CreateGenericTableSchema();
  TableSchema output_schema = CreateGenericTableSchema();

  SqlQuery query =
      CreateSqlQuery(input_schema, "SELECT * FROM t", output_schema);
  configure_request.mutable_configuration()->PackFrom(query);

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildRandomInt64SingleClientData(state.range(0)));
  TransformResponse transform_response;

  for (auto _ : state) {
    grpc::ClientContext transform_context;
    auto transform_status = stub_->Transform(
        &transform_context, transform_request, &transform_response);
  }
}

BENCHMARK_REGISTER_F(PipelineTransformBenchmark, BM_TransformSelectAll)
    ->Range(1, 5000);

class ComplexQueryPipelineTransformBenchmark
    : public PipelineTransformBenchmark {
 public:
  void SetUp(::benchmark::State& state) override {
    PipelineTransformBenchmark::SetUp(state);
    grpc::ClientContext configure_context;
    ConfigureAndAttestRequest configure_request;
    ConfigureAndAttestResponse configure_response;
    TableSchema input_schema = CreateGenericTableSchema();

    TableSchema output_schema = CreateComplexQueryOutputTableSchema();
    SqlQuery query = CreateSqlQuery(input_schema, kComplexQuery, output_schema);
    configure_request.mutable_configuration()->PackFrom(query);

    ASSERT_TRUE(stub_
                    ->ConfigureAndAttest(&configure_context, configure_request,
                                         &configure_response)
                    .ok());
  }
};

BENCHMARK_DEFINE_F(ComplexQueryPipelineTransformBenchmark,
                   BM_TransformComplexQuery)
(benchmark::State& state) {
  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildRandomInt64SingleClientData(state.range(0)));
  TransformResponse transform_response;

  for (auto _ : state) {
    grpc::ClientContext transform_context;
    auto transform_status = stub_->Transform(
        &transform_context, transform_request, &transform_response);
  }
}

BENCHMARK_REGISTER_F(ComplexQueryPipelineTransformBenchmark,
                     BM_TransformComplexQuery)
    ->Range(1, 5000);

BENCHMARK_DEFINE_F(ComplexQueryPipelineTransformBenchmark,
                   BM_TransformComplexQuery100Clients)
(benchmark::State& state) {
  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildRandomInt64SingleClientData(state.range(0)));
  TransformResponse transform_response;

  int num_clients = 100;
  for (auto _ : state) {
    for (int i = 0; i < num_clients; i++) {
      grpc::ClientContext transform_context;
      auto transform_status = stub_->Transform(
          &transform_context, transform_request, &transform_response);
    }
  }
}

BENCHMARK_REGISTER_F(ComplexQueryPipelineTransformBenchmark,
                     BM_TransformComplexQuery100Clients)
    ->Range(1, 5000);

// Use static SetUp and Teardown functions, since test fixtures setup and
// teardown don't work with multithreaded benchmarks.
static std::unique_ptr<SqlPipelineTransform> multi_thread_service_ = nullptr;
static std::unique_ptr<Server> multi_thread_server_ = nullptr;
static std::shared_ptr<grpc::Channel> multi_thread_channel_ = nullptr;
static std::unique_ptr<PipelineTransform::Stub> multi_thread_stub_ = nullptr;

static void DoSetup(const benchmark::State& state) {
  int port;
  multi_thread_service_ = std::make_unique<SqlPipelineTransform>();
  const std::string server_address = "[::1]:";
  ServerBuilder builder;
  builder.AddListeningPort(server_address + "0",
                           grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(multi_thread_service_.get());
  multi_thread_server_ = builder.BuildAndStart();
  multi_thread_channel_ =
      grpc::CreateChannel(server_address + std::to_string(port),
                          grpc::InsecureChannelCredentials());
  multi_thread_stub_ = PipelineTransform::NewStub(multi_thread_channel_);

  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  TableSchema input_schema = CreateGenericTableSchema();

  TableSchema output_schema = CreateComplexQueryOutputTableSchema();
  SqlQuery query = CreateSqlQuery(input_schema, kComplexQuery, output_schema);
  configure_request.mutable_configuration()->PackFrom(query);

  ASSERT_TRUE(multi_thread_stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());
}

static void DoTeardown(const benchmark::State& state) {
  multi_thread_stub_.reset(nullptr);
  multi_thread_channel_.reset();
  multi_thread_server_->Shutdown();
  multi_thread_server_.reset(nullptr);
  multi_thread_service_.reset(nullptr);
}

static void BM_ComplexQueryMultiThreadedTransform(benchmark::State& state) {
  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildRandomInt64SingleClientData(500));
  TransformResponse transform_response;
  for (auto _ : state) {
    grpc::ClientContext transform_context;
    auto transform_status = multi_thread_stub_->Transform(
        &transform_context, transform_request, &transform_response);
  }
}

BENCHMARK(BM_ComplexQueryMultiThreadedTransform)
    ->ThreadRange(1, 32)
    ->Setup(DoSetup)
    ->Teardown(DoTeardown);

}  // namespace

}  // namespace confidential_federated_compute::sql_server

BENCHMARK_MAIN();
