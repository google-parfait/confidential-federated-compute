#include "examples/sql_server/pipeline_transform_server.h"

#include "absl/log/check.h"
#include "examples/sql_server/sql_data.pb.h"
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

using ::fcp::aggregation::CheckpointBuilder;
using ::fcp::aggregation::CreateTestData;
using ::fcp::aggregation::DataType;
using ::fcp::aggregation::FederatedComputeCheckpointBuilderFactory;
using ::fcp::aggregation::Tensor;
using ::fcp::aggregation::TensorShape;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using grpc::Server;
using grpc::ServerBuilder;
using ::sql_data::DatabaseSchema;
using ::sql_data::TableSchema;
using ::testing::HasSubstr;
using ::testing::Test;

TableSchema CreateTableSchema(std::string name, std::string create_table_sql,
                              ColumnSchema column) {
  TableSchema schema;
  schema.set_name(name);
  schema.set_create_table_sql(create_table_sql);
  *(schema.add_column()) = column;
  return schema;
}

SqlQuery CreateSqlQuery(TableSchema input_table_schema, std::string raw_query,
                        TableSchema output_table_schema) {
  SqlQuery query;
  DatabaseSchema* input_schema = query.mutable_input_schema();
  DatabaseSchema* output_schema = query.mutable_output_schema();
  *(input_schema->add_table()) = input_table_schema;
  *(output_schema->add_table()) = output_table_schema;
  query.set_raw_sql(raw_query);
  return query;
}

ColumnSchema CreateColumnSchema(std::string name, ColumnSchema::DataType type) {
  ColumnSchema schema;
  schema.set_name(name);
  schema.set_type(type);
  return schema;
}

std::string BuildSingleInt64TensorCheckpoint(
    std::string column_name, std::initializer_list<uint64_t> input_values) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> t1 =
      Tensor::Create(DataType::DT_INT64,
                     TensorShape({static_cast<int64_t>(input_values.size())}),
                     CreateTestData<uint64_t>(input_values));
  CHECK_OK(t1);
  CHECK_OK(ckpt_builder->Add(column_name, *t1));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

class SqlPipelineTransformTest : public Test {
 public:
  void SetUp() override {
    int port;
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    channel_ = grpc::CreateChannel(server_address + std::to_string(port),
                                   grpc::InsecureChannelCredentials());
    stub_ = PipelineTransform::NewStub(channel_);
  }

  void TearDown() override { server_->Shutdown(); }

 protected:
  SqlPipelineTransform service_;
  std::unique_ptr<Server> server_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<PipelineTransform::Stub> stub_;
};

TEST_F(SqlPipelineTransformTest, InvalidConfigureAndAttestRequest) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("does not contain exactly one table schema"));
}

TEST_F(SqlPipelineTransformTest, ConfigureAndAttestMoreThanOnce) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  SqlQuery query = CreateSqlQuery(
      CreateTableSchema("input", "CREATE TABLE input (int_val INTEGER)",
                        CreateColumnSchema("int_val", ColumnSchema::INT64)),
      "SELECT SUM(int_val) AS int_sum FROM input",
      CreateTableSchema("output", "CREATE TABLE output (int_sum INTEGER)",
                        CreateColumnSchema("int_sum", ColumnSchema::INT64)));
  request.mutable_configuration()->PackFrom(query);

  ASSERT_TRUE(stub_->ConfigureAndAttest(&context, request, &response).ok());

  grpc::ClientContext second_context;
  auto status = stub_->ConfigureAndAttest(&second_context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("ConfigureAndAttest can only be called once"));
}

TEST_F(SqlPipelineTransformTest, ValidConfigureAndAttest) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  SqlQuery query = CreateSqlQuery(
      CreateTableSchema("input", "CREATE TABLE input (int_val INTEGER)",
                        CreateColumnSchema("int_val", ColumnSchema::INT64)),
      "SELECT SUM(int_val) AS int_sum FROM input",
      CreateTableSchema("output", "CREATE TABLE output (int_sum INTEGER)",
                        CreateColumnSchema("int_sum", ColumnSchema::INT64)));
  request.mutable_configuration()->PackFrom(query);

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_TRUE(status.ok());
}

TEST_F(SqlPipelineTransformTest, TransformExecutesSqlQuery) {
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "int_val";
  std::string output_col_name = "int_sum";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name, ColumnSchema::INT64));
  TableSchema output_schema = CreateTableSchema(
      "output",
      absl::StrFormat("CREATE TABLE output (%s INTEGER)", output_col_name),
      CreateColumnSchema(output_col_name, ColumnSchema::INT64));
  SqlQuery query =
      CreateSqlQuery(input_schema,
                     absl::StrFormat("SELECT SUM(%s) AS %s FROM input",
                                     input_col_name, output_col_name),
                     output_schema);
  configure_request.mutable_configuration()->PackFrom(query);

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildSingleInt64TensorCheckpoint(input_col_name, {1, 2}));
  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  auto transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok());
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());

  SqlData query_result;
  query_result.ParseFromString(
      transform_response.outputs(0).unencrypted_data());
  ASSERT_EQ(query_result.num_rows(), 1);
  ASSERT_TRUE(query_result.vector_data().vectors().contains(output_col_name));

  // The query sums the input column
  ExampleQueryResult_VectorData_Values result_values =
      query_result.vector_data().vectors().at(output_col_name);
  ASSERT_TRUE(result_values.has_int64_values());
  ASSERT_EQ(result_values.int64_values().value_size(), 1);
  ASSERT_EQ(result_values.int64_values().value(0), 3);
}

TEST_F(SqlPipelineTransformTest, TransformBeforeConfigureAndAttest) {
  grpc::ClientContext context;
  TransformRequest request;
  TransformResponse response;
  auto status = stub_->Transform(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("ConfigureAndAttest must be called before Transform"));
}
