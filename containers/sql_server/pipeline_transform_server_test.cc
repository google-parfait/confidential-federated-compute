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
#include "containers/sql_server/pipeline_transform_server.h"

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "proto/containers/orchestrator_crypto_mock.grpc.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

namespace confidential_federated_compute::sql_server {

namespace {

using ::fcp::client::ExampleQueryResult_VectorData;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::DatabaseSchema;
using ::fcp::confidentialcompute::GenerateNoncesRequest;
using ::fcp::confidentialcompute::GenerateNoncesResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::TableSchema;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_INT64;
using ::google::protobuf::RepeatedPtrField;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::StatusCode;
using ::oak::containers::v1::MockOrchestratorCryptoStub;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::HasSubstr;
using ::testing::SizeIs;
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
                        ColumnSchema output_column) {
  SqlQuery query;
  DatabaseSchema* input_schema = query.mutable_database_schema();
  *(input_schema->add_table()) = input_table_schema;
  *(query.add_output_columns()) = output_column;
  query.set_raw_sql(raw_query);
  return query;
}

ColumnSchema CreateColumnSchema(
    std::string name, ExampleQuerySpec_OutputVectorSpec_DataType type) {
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
  SqlPipelineTransformTest() {
    int port;
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    stub_ = PipelineTransform::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~SqlPipelineTransformTest() override { server_->Shutdown(); }

 protected:
  testing::NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub_;
  SqlPipelineTransform service_{&mock_crypto_stub_};
  std::unique_ptr<Server> server_;
  std::unique_ptr<PipelineTransform::Stub> stub_;
};

TEST_F(SqlPipelineTransformTest, InvalidConfigureAndAttestRequest) {
  grpc::ClientContext context;
  SqlQuery query;
  ConfigureAndAttestRequest request;
  request.mutable_configuration()->PackFrom(query);
  ConfigureAndAttestResponse response;
  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("does not contain exactly one table schema"));
}

TEST_F(SqlPipelineTransformTest, ConfigureAndAttestRequestWrongMessageType) {
  grpc::ClientContext context;
  google::protobuf::Value value;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  request.mutable_configuration()->PackFrom(value);

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(SqlPipelineTransformTest, ConfigureAndAttestMoreThanOnce) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  SqlQuery query = CreateSqlQuery(
      CreateTableSchema(
          "input", "CREATE TABLE input (int_val INTEGER)",
          CreateColumnSchema("int_val",
                             ExampleQuerySpec_OutputVectorSpec_DataType_INT64)),
      "SELECT SUM(int_val) AS int_sum FROM input",
      CreateColumnSchema("int_sum",
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  request.mutable_configuration()->PackFrom(query);

  ASSERT_TRUE(stub_->ConfigureAndAttest(&context, request, &response).ok());

  grpc::ClientContext second_context;
  auto status = stub_->ConfigureAndAttest(&second_context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("ConfigureAndAttest can only be called once"));
}

TEST_F(SqlPipelineTransformTest, ConfigureAndAttestInvalidTableSchema) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  SqlQuery query;
  DatabaseSchema* input_schema = query.mutable_database_schema();
  input_schema->add_table();
  request.mutable_configuration()->PackFrom(query);

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(),
              HasSubstr("SQL query input schema has no columns"));
}

TEST_F(SqlPipelineTransformTest, ValidConfigureAndAttest) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  SqlQuery query = CreateSqlQuery(
      CreateTableSchema(
          "input", "CREATE TABLE input (int_val INTEGER)",
          CreateColumnSchema("int_val",
                             ExampleQuerySpec_OutputVectorSpec_DataType_INT64)),
      "SELECT SUM(int_val) AS int_sum FROM input",
      CreateColumnSchema("int_sum",
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  request.mutable_configuration()->PackFrom(query);

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_TRUE(status.ok());
}

TEST_F(SqlPipelineTransformTest, ConfigureAndAttestPublicKeyHasDestBlobId) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  SqlQuery query = CreateSqlQuery(
      CreateTableSchema(
          "input", "CREATE TABLE input (int_val INTEGER)",
          CreateColumnSchema("int_val",
                             ExampleQuerySpec_OutputVectorSpec_DataType_INT64)),
      "SELECT SUM(int_val) AS int_sum FROM input",
      CreateColumnSchema("int_sum",
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  query.set_output_blob_id(42);
  request.mutable_configuration()->PackFrom(query);

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_TRUE(status.ok());

  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(response.public_key());
  ASSERT_TRUE(cwt.ok());
  ASSERT_EQ(cwt->config_properties.fields().at("dest").number_value(), 42);
}

TEST_F(SqlPipelineTransformTest, TransformExecutesSqlQuery) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "int_val";
  std::string output_col_name = "int_sum";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  SqlQuery query = CreateSqlQuery(
      input_schema,
      absl::StrFormat("SELECT SUM(%s) AS %s FROM input", input_col_name,
                      output_col_name),
      CreateColumnSchema(output_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  configure_request.mutable_configuration()->PackFrom(query);

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildSingleInt64TensorCheckpoint(input_col_name, {1, 2}));
  transform_request.mutable_inputs(0)->set_compression_type(
      Record::COMPRESSION_TYPE_NONE);
  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  auto transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok());
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());

  absl::Cord wire_format_result(
      transform_response.outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor(output_col_name);
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(*static_cast<const int64_t*>(col_values->data().data()), 3);
}

TEST_F(SqlPipelineTransformTest, TransformExecutesSqlQueryWithNoInputRows) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "no_rows";
  std::string output_col_name = "output_col";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  SqlQuery query = CreateSqlQuery(
      input_schema, "SELECT no_rows AS output_col FROM input",
      CreateColumnSchema(output_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  configure_request.mutable_configuration()->PackFrom(query);

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildSingleInt64TensorCheckpoint(input_col_name, {}));
  transform_request.mutable_inputs(0)->set_compression_type(
      Record::COMPRESSION_TYPE_NONE);
  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  auto transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);
  ASSERT_TRUE(transform_status.ok());
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());

  absl::Cord wire_format_result(
      transform_response.outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor(output_col_name);
  ASSERT_EQ(col_values->num_elements(), 0);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
}

TEST_F(SqlPipelineTransformTest, TransformDecryptsRecordAndExecutesSqlQuery) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "int_val";
  std::string output_col_name = "int_sum";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  SqlQuery query = CreateSqlQuery(
      input_schema,
      absl::StrFormat("SELECT SUM(%s) AS %s FROM input", input_col_name,
                      output_col_name),
      CreateColumnSchema(output_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  configure_request.mutable_configuration()->PackFrom(query);

  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();
  std::string recipient_public_key = configure_response.public_key();

  grpc::ClientContext nonces_context;
  GenerateNoncesRequest nonces_request;
  GenerateNoncesResponse nonces_response;
  nonces_request.set_nonces_count(1);
  grpc::Status generate_nonces_status =
      stub_->GenerateNonces(&nonces_context, nonces_request, &nonces_response);
  ASSERT_TRUE(generate_nonces_status.ok())
      << "GenerateNonces status code was: "
      << generate_nonces_status.error_code();
  ASSERT_THAT(nonces_response.nonces(), SizeIs(1));
  std::string nonce = nonces_response.nonces(0);

  std::string reencryption_public_key = "";
  std::string ciphertext_associated_data = "ciphertext associated data";
  std::string message =
      BuildSingleInt64TensorCheckpoint(input_col_name, {1, 2});
  absl::StatusOr<Record> rewrapped_record =
      crypto_test_utils::CreateRewrappedRecord(
          message, ciphertext_associated_data, recipient_public_key, nonce,
          reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();

  TransformRequest transform_request;
  transform_request.add_inputs()->CopyFrom(*rewrapped_record);

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok())
      << "Transform status code was: " << transform_status.error_code();
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());

  absl::Cord wire_format_result(
      transform_response.outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor(output_col_name);
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(*static_cast<const int64_t*>(col_values->data().data()), 3);
}

TEST_F(SqlPipelineTransformTest,
       BlobIdSetEncryptedInputsTransformEncryptsResult) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "int_val";
  std::string output_col_name = "int_sum";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  SqlQuery query = CreateSqlQuery(
      input_schema,
      absl::StrFormat("SELECT SUM(%s) AS %s FROM input", input_col_name,
                      output_col_name),
      CreateColumnSchema(output_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  query.set_output_blob_id(1);
  configure_request.mutable_configuration()->PackFrom(query);

  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();
  std::string recipient_public_key = configure_response.public_key();

  grpc::ClientContext nonces_context;
  GenerateNoncesRequest nonces_request;
  GenerateNoncesResponse nonces_response;
  nonces_request.set_nonces_count(1);
  grpc::Status generate_nonces_status =
      stub_->GenerateNonces(&nonces_context, nonces_request, &nonces_response);
  ASSERT_TRUE(generate_nonces_status.ok())
      << "GenerateNonces status code was: "
      << generate_nonces_status.error_code();
  ASSERT_THAT(nonces_response.nonces(), SizeIs(1));
  std::string nonce = nonces_response.nonces(0);

  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());
  std::string ciphertext_associated_data = "ciphertext associated data";
  std::string message =
      BuildSingleInt64TensorCheckpoint(input_col_name, {1, 2});
  absl::StatusOr<Record> rewrapped_record =
      crypto_test_utils::CreateRewrappedRecord(
          message, ciphertext_associated_data, recipient_public_key, nonce,
          *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();

  TransformRequest transform_request;
  transform_request.add_inputs()->CopyFrom(*rewrapped_record);

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok())
      << "Transform status code was: " << transform_status.error_code();
  ASSERT_EQ(transform_response.outputs_size(), 1);
  Record result = transform_response.outputs(0);
  ASSERT_TRUE(result.has_hpke_plus_aead_data());

  absl::StatusOr<std::string> decrypted_result = decryptor.Decrypt(
      result.hpke_plus_aead_data().ciphertext(),
      result.hpke_plus_aead_data().ciphertext_associated_data(),
      result.hpke_plus_aead_data().encrypted_symmetric_key(),
      result.hpke_plus_aead_data().ciphertext_associated_data(),
      result.hpke_plus_aead_data().encapsulated_public_key());
  ASSERT_TRUE(decrypted_result.ok()) << decrypted_result.status();

  absl::Cord wire_format_result(*decrypted_result);
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor(output_col_name);
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(*static_cast<const int64_t*>(col_values->data().data()), 3);
}

TEST_F(SqlPipelineTransformTest,
       BlobIdSetUnencryptedInputsTransformDoesNotEncryptOutputs) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "int_val";
  std::string output_col_name = "int_sum";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  SqlQuery query = CreateSqlQuery(
      input_schema,
      absl::StrFormat("SELECT SUM(%s) AS %s FROM input", input_col_name,
                      output_col_name),
      CreateColumnSchema(output_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  configure_request.mutable_configuration()->PackFrom(query);
  query.set_output_blob_id(2);
  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(
      BuildSingleInt64TensorCheckpoint(input_col_name, {1, 2}));
  transform_request.mutable_inputs(0)->set_compression_type(
      Record::COMPRESSION_TYPE_NONE);
  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  auto transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok());
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());
}

TEST_F(SqlPipelineTransformTest, TransformReturnsEmptyResponseForNoRecords) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "int_val";
  std::string output_col_name = "int_sum";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  SqlQuery query = CreateSqlQuery(
      input_schema,
      absl::StrFormat("SELECT SUM(%s) AS %s FROM input", input_col_name,
                      output_col_name),
      CreateColumnSchema(output_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  configure_request.mutable_configuration()->PackFrom(query);

  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();

  TransformRequest transform_request;

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok())
      << "Transform status code was: " << transform_status.error_code();
  ASSERT_EQ(transform_response.outputs_size(), 0);
}

TEST_F(SqlPipelineTransformTest, TransformReturnsNoOutputsIfDecryptionFails) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "int_val";
  std::string output_col_name = "int_sum";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  SqlQuery query = CreateSqlQuery(
      input_schema,
      absl::StrFormat("SELECT SUM(%s) AS %s FROM input", input_col_name,
                      output_col_name),
      CreateColumnSchema(output_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  configure_request.mutable_configuration()->PackFrom(query);

  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();

  std::string recipient_public_key = configure_response.public_key();

  grpc::ClientContext nonces_context;
  GenerateNoncesRequest nonces_request;
  GenerateNoncesResponse nonces_response;
  nonces_request.set_nonces_count(1);
  grpc::Status generate_nonces_status =
      stub_->GenerateNonces(&nonces_context, nonces_request, &nonces_response);
  ASSERT_TRUE(generate_nonces_status.ok())
      << "GenerateNonces status code was: "
      << generate_nonces_status.error_code();
  ASSERT_THAT(nonces_response.nonces(), SizeIs(1));
  std::string nonce = nonces_response.nonces(0);

  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());
  std::string ciphertext_associated_data = "ciphertext associated data";
  std::string message =
      BuildSingleInt64TensorCheckpoint(input_col_name, {1, 2});
  absl::StatusOr<Record> rewrapped_record =
      crypto_test_utils::CreateRewrappedRecord(
          message, ciphertext_associated_data, recipient_public_key, nonce,
          *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();

  TransformRequest transform_request;
  transform_request.add_inputs()->CopyFrom(*rewrapped_record);
  transform_request.mutable_inputs(0)
      ->mutable_hpke_plus_aead_data()
      ->set_ciphertext("invalid");

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok())
      << "Transform status code was: " << transform_status.error_code();
  ASSERT_EQ(transform_response.outputs_size(), 0);
  ASSERT_EQ(transform_response.num_ignored_inputs(), 1);
}

TEST_F(SqlPipelineTransformTest,
       TransformReturnsNoOutputsIfCheckpointUnparseable) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "int_val";
  std::string output_col_name = "int_sum";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  SqlQuery query = CreateSqlQuery(
      input_schema,
      absl::StrFormat("SELECT SUM(%s) AS %s FROM input", input_col_name,
                      output_col_name),
      CreateColumnSchema(output_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  configure_request.mutable_configuration()->PackFrom(query);

  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data("invalid_checkpoint");
  transform_request.mutable_inputs(0)->set_compression_type(
      Record::COMPRESSION_TYPE_NONE);

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok())
      << "Transform status code was: " << transform_status.error_code();
  ASSERT_EQ(transform_response.outputs_size(), 0);
  ASSERT_EQ(transform_response.num_ignored_inputs(), 1);
}

TEST_F(SqlPipelineTransformTest,
       TransformReturnsNoOutputsIfCheckpointHasWrongColumn) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "int_val";
  std::string output_col_name = "int_sum";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  SqlQuery query = CreateSqlQuery(
      input_schema,
      absl::StrFormat("SELECT SUM(%s) AS %s FROM input", input_col_name,
                      output_col_name),
      CreateColumnSchema(output_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  configure_request.mutable_configuration()->PackFrom(query);

  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();

  std::string checkpoint =
      BuildSingleInt64TensorCheckpoint("wrong_col_name", {1, 2});
  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(checkpoint);
  transform_request.mutable_inputs(0)->set_compression_type(
      Record::COMPRESSION_TYPE_NONE);

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok())
      << "Transform status code was: " << transform_status.error_code();
  ASSERT_EQ(transform_response.outputs_size(), 0);
  ASSERT_EQ(transform_response.num_ignored_inputs(), 1);
}

TEST_F(SqlPipelineTransformTest, TransformFailsForMultipleRecords) {
  FederatedComputeCheckpointParserFactory parser_factory;
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  std::string input_col_name = "int_val";
  std::string output_col_name = "int_sum";
  TableSchema input_schema = CreateTableSchema(
      "input",
      absl::StrFormat("CREATE TABLE input (%s INTEGER)", input_col_name),
      CreateColumnSchema(input_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  SqlQuery query = CreateSqlQuery(
      input_schema,
      absl::StrFormat("SELECT SUM(%s) AS %s FROM input", input_col_name,
                      output_col_name),
      CreateColumnSchema(output_col_name,
                         ExampleQuerySpec_OutputVectorSpec_DataType_INT64));
  configure_request.mutable_configuration()->PackFrom(query);

  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();

  std::string message_1 =
      BuildSingleInt64TensorCheckpoint(input_col_name, {1, 2});
  std::string message_2 =
      BuildSingleInt64TensorCheckpoint(input_col_name, {3, 4, 5});

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data(message_1);
  transform_request.add_inputs()->set_unencrypted_data(message_2);

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_EQ(transform_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(transform_status.error_message(),
              HasSubstr("Transform requires at most one `Record` per request"));
}

TEST_F(SqlPipelineTransformTest, TransformBeforeConfigureAndAttest) {
  grpc::ClientContext context;
  TransformRequest request;
  request.add_inputs();
  TransformResponse response;
  auto status = stub_->Transform(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("ConfigureAndAttest must be called before Transform"));
}

TEST_F(SqlPipelineTransformTest, GenerateNoncesBeforeConfigureAndAttest) {
  grpc::ClientContext nonces_context;
  GenerateNoncesRequest nonces_request;
  GenerateNoncesResponse nonces_response;
  nonces_request.set_nonces_count(1);
  auto status =
      stub_->GenerateNonces(&nonces_context, nonces_request, &nonces_response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(
      status.error_message(),
      HasSubstr("ConfigureAndAttest must be called before GenerateNonces"));
}

}  // namespace

}  // namespace confidential_federated_compute::sql_server
