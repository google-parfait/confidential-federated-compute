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
#include "containers/fed_sql/confidential_transform_server.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "cc/crypto/client_encryptor.h"
#include "cc/crypto/encryption_key.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "containers/fed_sql/budget.pb.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/fed_sql/testing/mocks.h"
#include "containers/fed_sql/testing/test_utils.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/private_state.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "google/protobuf/struct.pb.h"
#include "google/rpc/code.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::crypto_test_utils::MockSigningKeyHandle;
using ::confidential_federated_compute::fed_sql::testing::
    BuildFedSqlGroupByCheckpoint;
using ::confidential_federated_compute::fed_sql::testing::MockInferenceModel;
using ::fcp::base::FromGrpcStatus;
using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::kPrivateStateConfigId;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::NonceAndCounter;
using ::fcp::confidential_compute::NonceGenerator;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::AggCoreAggregationType;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_MERGE;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigurationMetadata;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::DatabaseSchema;
using ::fcp::confidentialcompute::FedSqlContainerConfigConstraints;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerInitializeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::TableSchema;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::RepeatedPtrField;
using ::google::rpc::Code;
using ::grpc::ClientWriter;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::oak::crypto::ClientEncryptor;
using ::oak::crypto::EncryptionKeyProvider;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::Configuration;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::AnyOf;
using ::testing::ByMove;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Not;
using ::testing::Return;
using ::testing::SizeIs;
using ::testing::Test;
using ::testing::UnorderedElementsAre;

inline constexpr int kMaxNumSessions = 8;
inline constexpr int kSerializeOutputNodeId = 1;
inline constexpr int kReportOutputNodeId = 2;

TableSchema CreateTableSchema(std::string name, std::string create_table_sql,
                              std::vector<ColumnSchema> columns) {
  TableSchema schema;
  schema.set_name(name);
  schema.set_create_table_sql(create_table_sql);
  schema.mutable_column()->Add(columns.begin(), columns.end());
  return schema;
}

SqlQuery CreateSqlQuery(TableSchema input_table_schema, std::string raw_query,
                        std::vector<ColumnSchema> output_columns) {
  SqlQuery query;
  DatabaseSchema* input_schema = query.mutable_database_schema();
  *(input_schema->add_table()) = input_table_schema;
  query.mutable_output_columns()->Add(output_columns.begin(),
                                      output_columns.end());
  query.set_raw_sql(raw_query);
  return query;
}

ColumnSchema CreateColumnSchema(
    std::string name, google::internal::federated::plan::DataType type) {
  ColumnSchema schema;
  schema.set_name(name);
  schema.set_type(type);
  return schema;
}

Configuration DefaultConfiguration() {
  return PARSE_TEXT_PROTO(R"pb(
    intrinsic_configs: {
      intrinsic_uri: "fedsql_group_by"
      intrinsic_args {
        input_tensor {
          name: "key"
          dtype: DT_INT64
          shape { dim_sizes: -1 }
        }
      }
      output_tensors {
        name: "key_out"
        dtype: DT_INT64
        shape { dim_sizes: -1 }
      }
      inner_intrinsics {
        intrinsic_uri: "GoogleSQL:sum"
        intrinsic_args {
          input_tensor {
            name: "val"
            dtype: DT_INT64
            shape {}
          }
        }
        output_tensors {
          name: "val_out"
          dtype: DT_INT64
          shape {}
        }
      }
    }
  )pb");
}

SqlQuery DefaultSqlQuery() {
  TableSchema schema = CreateTableSchema(
      "input", "CREATE TABLE input (key INTEGER, val INTEGER)",
      {CreateColumnSchema("key", google::internal::federated::plan::INT64),
       CreateColumnSchema("val", google::internal::federated::plan::INT64)});
  return CreateSqlQuery(
      schema, "SELECT key, val * 2 AS val FROM input",
      {CreateColumnSchema("key", google::internal::federated::plan::INT64),
       CreateColumnSchema("val", google::internal::federated::plan::INT64)});
}

SessionRequest CreateDefaultWriteRequest(AggCoreAggregationType agg_type,
                                         std::string data) {
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  metadata.set_total_size_bytes(data.size());
  FedSqlContainerWriteConfiguration config;
  config.set_type(agg_type);
  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->mutable_first_request_configuration()->PackFrom(config);
  write_request->set_commit(true);
  write_request->set_data(data);
  return request;
}

std::string ReadFileContent(std::string file_path) {
  std::ifstream temp_file(file_path);
  std::stringstream buffer;
  buffer << temp_file.rdbuf();
  temp_file.close();
  return buffer.str();
}

// Write the InitializeRequest to the client stream and then close
// the stream, returning the status of Finish.
absl::Status WriteInitializeRequest(
    std::unique_ptr<ClientWriter<StreamInitializeRequest>> stream,
    InitializeRequest request) {
  StreamInitializeRequest stream_request;
  *stream_request.mutable_initialize_request() = std::move(request);
  if (!stream->Write(stream_request)) {
    return absl::AbortedError("Write to StreamInitialize failed.");
  }
  if (!stream->WritesDone()) {
    return absl::AbortedError("WritesDone to StreamInitialize failed.");
  }
  return FromGrpcStatus(stream->Finish());
}

bool WritePipelinePrivateState(ClientWriter<StreamInitializeRequest>* stream,
                               const std::string& state) {
  StreamInitializeRequest stream_request;
  auto* write_configuration = stream_request.mutable_write_configuration();
  write_configuration->set_commit(true);
  write_configuration->set_data(state);
  auto* metadata = write_configuration->mutable_first_request_metadata();
  metadata->set_configuration_id(kPrivateStateConfigId);
  metadata->set_total_size_bytes(state.size());
  return stream->Write(stream_request);
}

class FedSqlServerTest : public Test {
 public:
  FedSqlServerTest() {
    int port;
    const std::string server_address = "[::1]:";

    ON_CALL(*mock_inference_model_, BuildGemmaModel).WillByDefault(Return());
    ON_CALL(*mock_inference_model_, RunGemmaInference)
        .WillByDefault(Return("topic_value"));
    auto encryption_key_handle = std::make_unique<EncryptionKeyProvider>(
        EncryptionKeyProvider::Create().value());
    oak_client_encryptor_ =
        ClientEncryptor::Create(encryption_key_handle->GetSerializedPublicKey())
            .value();
    service_ = std::make_unique<FedSqlConfidentialTransform>(
        std::make_unique<NiceMock<MockSigningKeyHandle>>(),
        std::move(encryption_key_handle));

    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    stub_ = ConfidentialTransform::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~FedSqlServerTest() override {
    server_->Shutdown();
    // Clean up any temp files created by the server.
    for (auto& de : std::filesystem::directory_iterator("/tmp")) {
      std::filesystem::remove_all(de.path());
    }
  }

 protected:
  InferenceInitializeConfiguration DefaultInferenceConfig() const;
  FedSqlContainerInitializeConfiguration DefaultFedSqlDpContainerConfig() const;
  // Returns the default BlobMetadata
  BlobMetadata DefaultBlobMetadata() const;

  std::shared_ptr<NiceMock<MockInferenceModel>> mock_inference_model_ =
      std::make_shared<NiceMock<MockInferenceModel>>();
  std::unique_ptr<FedSqlConfidentialTransform> service_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
  std::unique_ptr<ClientEncryptor> oak_client_encryptor_;
};

InferenceInitializeConfiguration FedSqlServerTest::DefaultInferenceConfig()
    const {
  return PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_name: "transcript"
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
      gemma_config {
        tokenizer_file: "/path/to/tokenizer"
        model_weight_file: "/path/to/model_weight"
        model: GEMMA_2B
        model_training: GEMMA_IT
        tensor_type: GEMMA_SFP
      }
    }
    gemma_init_config {
      tokenizer_configuration_id: "gemma_tokenizer_id"
      model_weight_configuration_id: "gemma_model_weight_id"
    }
  )pb");
}

FedSqlContainerInitializeConfiguration
FedSqlServerTest::DefaultFedSqlDpContainerConfig() const {
  return PARSE_TEXT_PROTO(R"pb(
    agg_configuration {
      intrinsic_configs: {
        intrinsic_uri: "fedsql_dp_group_by"
        intrinsic_args { parameter { dtype: DT_DOUBLE double_val: 1.1 } }
        intrinsic_args { parameter { dtype: DT_DOUBLE double_val: 0.01 } }
        intrinsic_args { parameter { dtype: DT_INT64 int64_val: 0 } }
        output_tensors {
          name: "key_out"
          dtype: DT_INT64
          shape { dim_sizes: -1 }
        }
        inner_intrinsics {
          intrinsic_uri: "GoogleSQL:sum"
          intrinsic_args {
            input_tensor {
              name: "val"
              dtype: DT_INT64
              shape {}
            }
          }
          output_tensors {
            name: "val_out"
            dtype: DT_INT64
            shape {}
          }
        }
      }
    }
    serialize_output_access_policy_node_id: 42
    report_output_access_policy_node_id: 7
  )pb");
}

TEST_F(FedSqlServerTest, ValidStreamInitializeWithoutInferenceConfigs) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  request.mutable_configuration()->PackFrom(init_config);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      IsOk());

  // Inference files shouldn't exist because no write_configuration is provided.
  ASSERT_FALSE(std::filesystem::exists("/tmp/write_configuration_1"));
}

TEST_F(FedSqlServerTest, InvalidStreamInitializeRequest) {
  grpc::ClientContext context;
  FedSqlContainerInitializeConfiguration invalid_config;
  invalid_config.mutable_agg_configuration()
      ->add_intrinsic_configs()
      ->set_intrinsic_uri("BAD URI");
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(invalid_config);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("is not a supported intrinsic_uri")));
}

TEST_F(FedSqlServerTest, StreamInitializeRequestNoIntrinsicConfigs) {
  grpc::ClientContext context;
  FedSqlContainerInitializeConfiguration invalid_config;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(invalid_config);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Configuration must have exactly one IntrinsicConfig")));
}

TEST_F(FedSqlServerTest, FedSqlDpGroupByInvalidParametersStreamInitialize) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  FedSqlContainerInitializeConfiguration init_config = PARSE_TEXT_PROTO(R"pb(
    agg_configuration {
      intrinsic_configs: {
        intrinsic_uri: "fedsql_dp_group_by"
        intrinsic_args { parameter { dtype: DT_INT64 int64_val: 42 } }
        intrinsic_args { parameter { dtype: DT_DOUBLE double_val: 0.01 } }
        intrinsic_args { parameter { dtype: DT_INT64 int64_val: 42 } }
        output_tensors {
          name: "key_out"
          dtype: DT_INT64
          shape { dim_sizes: -1 }
        }
        inner_intrinsics {
          intrinsic_uri: "GoogleSQL:sum"
          intrinsic_args {
            input_tensor {
              name: "val"
              dtype: DT_INT64
              shape {}
            }
          }
          output_tensors {
            name: "val_out"
            dtype: DT_INT64
            shape {}
          }
        }
      }
    }
  )pb");
  request.mutable_configuration()->PackFrom(init_config);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("must both have type DT_DOUBLE")));
}

TEST_F(FedSqlServerTest, MultipleTopLevelIntrinsicsStreamInitialize) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  FedSqlContainerInitializeConfiguration init_config = PARSE_TEXT_PROTO(R"pb(
    agg_configuration {
      intrinsic_configs: {
        intrinsic_uri: "federated_sum"
        output_tensors {
          name: "key_out"
          dtype: DT_INT64
          shape { dim_sizes: -1 }
        }
      }
      intrinsic_configs: {
        intrinsic_uri: "federated_sum"
        output_tensors {
          name: "key_out"
          dtype: DT_INT64
          shape { dim_sizes: -1 }
        }
      }
    }
  )pb");
  request.mutable_configuration()->PackFrom(init_config);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Configuration must have exactly one IntrinsicConfig")));
}

TEST_F(FedSqlServerTest, StreamInitializeMoreThanOnce) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  request.mutable_configuration()->PackFrom(init_config);

  EXPECT_THAT(WriteInitializeRequest(
                  stub_->StreamInitialize(&context, &response), request),
              IsOk());

  grpc::ClientContext second_context;
  EXPECT_THAT(WriteInitializeRequest(
                  stub_->StreamInitialize(&second_context, &response),
                  std::move(request)),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("SetIntrinsics can only be called once")));
}

TEST_F(FedSqlServerTest,
       FedSqlDpGroupByStreamInitializeGeneratesConfigProperties) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(DefaultFedSqlDpContainerConfig());

  ASSERT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      IsOk());

  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(response.public_key());
  ASSERT_THAT(cwt, IsOk());
  google::protobuf::Struct config_properties;
  ASSERT_TRUE(config_properties.ParseFromString(cwt->config_properties));
  ASSERT_EQ(config_properties.fields().at("intrinsic_uri").string_value(),
            "fedsql_dp_group_by");
  ASSERT_EQ(config_properties.fields().at("epsilon").number_value(), 1.1);
  ASSERT_EQ(config_properties.fields().at("delta").number_value(), 0.01);
  ASSERT_EQ(config_properties.fields().at("serialize_dest").number_value(), 42);
  ASSERT_EQ(config_properties.fields().at("report_dest").number_value(), 7);
}

TEST_F(FedSqlServerTest,
       FedSqlDpAggregatorBundleStreamInitializeGeneratesConfigProperties) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  FedSqlContainerInitializeConfiguration init_config = PARSE_TEXT_PROTO(R"pb(
    agg_configuration {
      intrinsic_configs: {
        intrinsic_uri: "differential_privacy_tensor_aggregator_bundle"
        intrinsic_args { parameter { dtype: DT_DOUBLE double_val: 1.1 } }
        intrinsic_args { parameter { dtype: DT_DOUBLE double_val: 0.01 } }
        output_tensors {
          name: "key_out"
          dtype: DT_INT64
          shape { dim_sizes: -1 }
        }
        inner_intrinsics {
          intrinsic_uri: "GoogleSQL:$differential_privacy_percentile_cont"
          intrinsic_args {
            input_tensor {
              name: "val"
              dtype: DT_INT64
              shape {}
            }
          }
          intrinsic_args {
            parameter {
              dtype: DT_DOUBLE
              shape {}
              double_val: 0.83
            }
          }
          output_tensors {
            name: "val_out"
            dtype: DT_INT64
            shape {}
          }
        }
      }
    }
    serialize_output_access_policy_node_id: 42
    report_output_access_policy_node_id: 7
  )pb");
  request.mutable_configuration()->PackFrom(init_config);

  ASSERT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      IsOk());

  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(response.public_key());
  ASSERT_THAT(cwt, IsOk());
  google::protobuf::Struct config_properties;
  ASSERT_TRUE(config_properties.ParseFromString(cwt->config_properties));
  ASSERT_EQ(config_properties.fields().at("intrinsic_uri").string_value(),
            "differential_privacy_tensor_aggregator_bundle");
  ASSERT_EQ(config_properties.fields().at("epsilon").number_value(), 1.1);
  ASSERT_EQ(config_properties.fields().at("delta").number_value(), 0.01);
  ASSERT_EQ(config_properties.fields().at("serialize_dest").number_value(), 42);
  ASSERT_EQ(config_properties.fields().at("report_dest").number_value(), 7);
}

TEST_F(FedSqlServerTest, StreamInitializeWithKmsSuccess) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(DefaultFedSqlDpContainerConfig());

  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    epsilon: 1.1
    delta: 0.01
    intrinsic_uri: "fedsql_dp_group_by"
    access_budget { times: 5 }
  )pb");
  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(config_constraints);
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;
  auto writer = stub_->StreamInitialize(&context, &response);

  BudgetState budget_state =
      PARSE_TEXT_PROTO(R"pb(buckets { key: "foo" budget: 1 })pb");
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(),
                                        budget_state.SerializeAsString()));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());
}

TEST_F(FedSqlServerTest, StreamInitializeWithKmsNoDpConfig) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  request.mutable_configuration()->PackFrom(init_config);

  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    intrinsic_uri: "fedsql_group_by")pb");
  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(config_constraints);
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());
}

TEST_F(FedSqlServerTest, StreamInitializeWithKmsInvalidPrivateState) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  request.mutable_configuration()->PackFrom(init_config);

  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    intrinsic_uri: "fedsql_group_by")pb");
  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(config_constraints);
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), "invalid private state"));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(FedSqlServerTest, StreamInitializeWithKmsInvalidUri) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(DefaultFedSqlDpContainerConfig());

  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    epsilon: 1.1
    delta: 0.01
    intrinsic_uri: "my_intrinsic_uri")pb");
  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(config_constraints);
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Invalid intrinsic URI for DP configuration.")));
}

TEST_F(FedSqlServerTest, StreamInitializeWithKmsInvalidEpsilon) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(DefaultFedSqlDpContainerConfig());

  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    epsilon: 1.2
    delta: 0.01
    intrinsic_uri: "fedsql_dp_group_by")pb");
  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(config_constraints);
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Epsilon value does not match the expected value.")));
}

TEST_F(FedSqlServerTest, StreamInitializeWithKmsInvalidDelta) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(DefaultFedSqlDpContainerConfig());

  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    epsilon: 1.1
    delta: 2.3
    intrinsic_uri: "fedsql_dp_group_by")pb");
  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(config_constraints);
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Delta value does not match the expected value.")));
}

TEST_F(FedSqlServerTest, StreamInitializeWithKmsInvalidConfigConstraints) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(DefaultFedSqlDpContainerConfig());

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  google::protobuf::Value value;
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(value);
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("FedSqlContainerConfigConstraints cannot be unpacked.")));
}

TEST_F(FedSqlServerTest, StreamInitializeRequestWrongMessageType) {
  grpc::ClientContext context;
  google::protobuf::Value value;
  InitializeRequest request;
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(value);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Configuration cannot be unpacked.")));
}

TEST_F(FedSqlServerTest, ValidStreamInitializeWithInferenceConfigs) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest initialize_request;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  initialize_request.mutable_configuration()->PackFrom(init_config);

  // Set tokenizer data blob.
  std::string expected_tokenizer_content = "test tokenizer content";
  StreamInitializeRequest tokenizer_write_config;
  ConfigurationMetadata* tokenizer_metadata =
      tokenizer_write_config.mutable_write_configuration()
          ->mutable_first_request_metadata();
  tokenizer_metadata->set_configuration_id("gemma_tokenizer_id");
  tokenizer_write_config.mutable_write_configuration()->set_commit(true);
  absl::Cord tokenizer_content(expected_tokenizer_content);
  tokenizer_metadata->set_total_size_bytes(tokenizer_content.size());
  std::string tokenizer_content_string;
  absl::CopyCordToString(tokenizer_content, &tokenizer_content_string);
  tokenizer_write_config.mutable_write_configuration()->set_data(
      tokenizer_content_string);

  // Set up model weight data blob.
  // Reuse data for the first and second WriteConfigurationRequest for the model
  // weight blob.
  std::string expected_model_weight_content = "test first model weight content";
  absl::Cord model_weight_content(expected_model_weight_content);
  std::string model_weight_content_string;
  absl::CopyCordToString(model_weight_content, &model_weight_content_string);

  // Set the first WriteConfigurationRequest for the model weight data blob.
  StreamInitializeRequest first_model_weight_write_config;
  ConfigurationMetadata* first_model_weight_metadata =
      first_model_weight_write_config.mutable_write_configuration()
          ->mutable_first_request_metadata();
  // model_weight_content is sent twice.
  first_model_weight_metadata->set_total_size_bytes(
      model_weight_content.size() * 2);
  first_model_weight_metadata->set_configuration_id("gemma_model_weight_id");

  first_model_weight_write_config.mutable_write_configuration()->set_data(
      model_weight_content_string);

  // Set the second WriteConfigurationRequest for the model weight data blob.
  StreamInitializeRequest second_model_weight_write_config;
  second_model_weight_write_config.mutable_write_configuration()->set_commit(
      true);
  second_model_weight_write_config.mutable_write_configuration()->set_data(
      model_weight_content_string);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(tokenizer_write_config));
  ASSERT_TRUE(writer->Write(first_model_weight_write_config));
  ASSERT_TRUE(writer->Write(second_model_weight_write_config));

  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      IsOk());

  ASSERT_TRUE(std::filesystem::exists("/tmp/write_configuration_1"));
  std::string tokenizer_file_content =
      ReadFileContent("/tmp/write_configuration_1");
  ASSERT_EQ(expected_tokenizer_content, tokenizer_file_content);

  ASSERT_TRUE(std::filesystem::exists("/tmp/write_configuration_2"));
  std::string model_weight_file_content =
      ReadFileContent("/tmp/write_configuration_2");
  ASSERT_EQ(absl::StrCat(expected_model_weight_content,
                         expected_model_weight_content),
            model_weight_file_content);
}

TEST_F(FedSqlServerTest, StreamInitializeMissingModelInitConfig) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest request;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  *init_config.mutable_inference_init_config() = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_name: "transcript"
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
      gemma_config {
        tokenizer_file: "/path/to/tokenizer"
        model_weight_file: "/path/to/model_weight"
        model: GEMMA_2B
        model_training: GEMMA_IT
        tensor_type: GEMMA_SFP
      }
    }
  )pb");
  request.mutable_configuration()->PackFrom(init_config);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("model_init_config must be set.")));
}

TEST_F(FedSqlServerTest, StreamInitializeMissingModelConfig) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest request;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  *init_config.mutable_inference_init_config() = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_name: "transcript"
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
    }
    gemma_init_config {
      tokenizer_configuration_id: "gemma_tokenizer_id"
      model_weight_configuration_id: "gemma_model_weight_id"
    }
  )pb");
  request.mutable_configuration()->PackFrom(init_config);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("model_config must be set.")));
}

TEST_F(FedSqlServerTest, StreamInitializeMissingInferenceLogic) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest request;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  *init_config.mutable_inference_init_config() = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_name: "transcript"
          output_column_name: "topic"
        }
      }
      gemma_config {
        tokenizer_file: "/path/to/tokenizer"
        model_weight_file: "/path/to/model_weight"
        model: GEMMA_2B
        model_training: GEMMA_IT
        tensor_type: GEMMA_SFP
      }
    }
    gemma_init_config {
      tokenizer_configuration_id: "gemma_tokenizer_id"
      model_weight_configuration_id: "gemma_model_weight_id"
    }
  )pb");
  request.mutable_configuration()->PackFrom(init_config);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("inference_task.inference_logic must be set for all "
                         "inference tasks.")));
}

TEST_F(FedSqlServerTest,
       StreamInitializeWriteConfigurationRequestNotCommitted) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest initialize_request;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  initialize_request.mutable_configuration()->PackFrom(init_config);

  StreamInitializeRequest write_configuration;
  ConfigurationMetadata* metadata =
      write_configuration.mutable_write_configuration()
          ->mutable_first_request_metadata();
  metadata->set_configuration_id("gemma_tokenizer_id");
  metadata->set_total_size_bytes(0);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(write_configuration));

  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Data blob with configuration_id gemma_tokenizer_id is "
                    "not committed.")));
}

TEST_F(FedSqlServerTest, StreamInitializeInvalidGemmaTokenizerConfigurationId) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest initialize_request;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  initialize_request.mutable_configuration()->PackFrom(init_config);

  StreamInitializeRequest write_configuration;
  ConfigurationMetadata* metadata =
      write_configuration.mutable_write_configuration()
          ->mutable_first_request_metadata();
  metadata->set_configuration_id("invalid_configuration_id");
  metadata->set_total_size_bytes(0);
  write_configuration.mutable_write_configuration()->set_commit(true);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(write_configuration));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected Gemma tokenizer configuration id "
              "gemma_tokenizer_id is missing in WriteConfigurationRequest.")));
}

TEST_F(FedSqlServerTest,
       StreamInitializeInvalidGemmaModelWeightConfigurationId) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest initialize_request;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  initialize_request.mutable_configuration()->PackFrom(init_config);

  StreamInitializeRequest tokenizer_write_config;
  ConfigurationMetadata* tokenizer_metadata =
      tokenizer_write_config.mutable_write_configuration()
          ->mutable_first_request_metadata();
  tokenizer_metadata->set_configuration_id("gemma_tokenizer_id");
  tokenizer_metadata->set_total_size_bytes(0);
  tokenizer_write_config.mutable_write_configuration()->set_commit(true);

  StreamInitializeRequest model_weight_write_config;
  ConfigurationMetadata* model_weight_metadata =
      model_weight_write_config.mutable_write_configuration()
          ->mutable_first_request_metadata();
  model_weight_metadata->set_configuration_id("invalid_gemma_model_weight_id");
  model_weight_metadata->set_total_size_bytes(0);
  model_weight_write_config.mutable_write_configuration()->set_commit(true);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(tokenizer_write_config));
  ASSERT_TRUE(writer->Write(model_weight_write_config));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected Gemma model weight configuration id "
                         "gemma_model_weight_id is missing in "
                         "WriteConfigurationRequest.")));
}

TEST_F(FedSqlServerTest, StreamInitializeDuplicatedConfigurationId) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest initialize_request;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  initialize_request.mutable_configuration()->PackFrom(init_config);

  StreamInitializeRequest tokenizer_write_config;
  ConfigurationMetadata* tokenizer_metadata =
      tokenizer_write_config.mutable_write_configuration()
          ->mutable_first_request_metadata();
  tokenizer_metadata->set_configuration_id("gemma_tokenizer_id");
  tokenizer_metadata->set_total_size_bytes(0);
  tokenizer_write_config.mutable_write_configuration()->set_commit(true);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(tokenizer_write_config));
  ASSERT_TRUE(writer->Write(tokenizer_write_config));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Duplicated configuration_id found in "
                         "WriteConfigurationRequest")));
}

TEST_F(FedSqlServerTest, StreamInitializeInconsistentTotalSizeBytes) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest initialize_request;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  initialize_request.mutable_configuration()->PackFrom(init_config);

  // Set tokenizer data blob.
  StreamInitializeRequest write_configuration;
  ConfigurationMetadata* metadata =
      write_configuration.mutable_write_configuration()
          ->mutable_first_request_metadata();
  metadata->set_configuration_id("gemma_tokenizer_id");
  write_configuration.mutable_write_configuration()->set_commit(true);

  std::string tokenizer_content = "fake tokenizer content";
  // Set total_size_bytes to an incorrect value
  metadata->set_total_size_bytes(9999);
  write_configuration.mutable_write_configuration()->set_data(
      tokenizer_content);
  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(write_configuration));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("The total size of the data blob does not match "
                         "expected size.")));
}

TEST_F(FedSqlServerTest, SessionBeforeInitialize) {
  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->set_chunk_size(1000);
  configure_request.mutable_configure()->mutable_configuration();

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_FALSE(stream->Read(&configure_response));
  EXPECT_THAT(FromGrpcStatus(stream->Finish()),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Initialize must be called before Session")));
}

TEST_F(FedSqlServerTest, CreateSessionWithKmsEnabledSucceeds) {
  grpc::ClientContext context;
  InitializeRequest request;
  request.set_max_num_sessions(kMaxNumSessions);
  InitializeResponse response;
  FedSqlContainerInitializeConfiguration init_config;
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  request.mutable_configuration()->PackFrom(init_config);
  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    intrinsic_uri: "fedsql_group_by")pb");

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "merge_encryption_key";
  *protected_response.add_result_encryption_keys() = "report_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(config_constraints);
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());

  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->set_chunk_size(1000);
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
}

std::string BuildFedSqlGroupByStringKeyCheckpoint(
    std::initializer_list<absl::string_view> key_col_values,
    std::string key_col_name) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> key =
      Tensor::Create(DataType::DT_STRING,
                     TensorShape({static_cast<int64_t>(key_col_values.size())}),
                     CreateTestData<absl::string_view>(key_col_values));
  CHECK_OK(key);
  CHECK_OK(ckpt_builder->Add(key_col_name, *key));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

std::string BuildSensitiveGroupByCheckpoint(
    std::initializer_list<absl::string_view> key_col_values,
    std::initializer_list<uint64_t> val_col_values) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> key =
      Tensor::Create(DataType::DT_STRING,
                     TensorShape({static_cast<int64_t>(key_col_values.size())}),
                     CreateTestData<absl::string_view>(key_col_values));
  absl::StatusOr<Tensor> val =
      Tensor::Create(DataType::DT_INT64,
                     TensorShape({static_cast<int64_t>(val_col_values.size())}),
                     CreateTestData<uint64_t>(val_col_values));
  CHECK_OK(key);
  CHECK_OK(val);
  CHECK_OK(ckpt_builder->Add("SENSITIVE_key", *key));
  CHECK_OK(ckpt_builder->Add("val", *val));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

TEST_F(FedSqlServerTest, SensitiveColumnsAreHashed) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  FedSqlContainerInitializeConfiguration init_config = PARSE_TEXT_PROTO(R"pb(
    agg_configuration {
      intrinsic_configs: {
        intrinsic_uri: "fedsql_group_by"
        intrinsic_args {
          input_tensor {
            name: "SENSITIVE_key"
            dtype: DT_STRING
            shape { dim_sizes: -1 }
          }
        }
        output_tensors {
          name: "SENSITIVE_key_out"
          dtype: DT_STRING
          shape { dim_sizes: -1 }
        }
        inner_intrinsics {
          intrinsic_uri: "GoogleSQL:sum"
          intrinsic_args {
            input_tensor {
              name: "val"
              dtype: DT_INT64
              shape {}
            }
          }
          output_tensors {
            name: "val_out"
            dtype: DT_INT64
            shape {}
          }
        }
      }
    }
  )pb");
  request.mutable_configuration()->PackFrom(init_config);
  request.set_max_num_sessions(kMaxNumSessions);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      IsOk());

  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  TableSchema schema = CreateTableSchema(
      "input", "CREATE TABLE input (SENSITIVE_key STRING, val INTEGER)",
      {CreateColumnSchema("SENSITIVE_key",
                          google::internal::federated::plan::STRING),
       CreateColumnSchema("val", google::internal::federated::plan::INT64)});
  SqlQuery query = CreateSqlQuery(
      schema, "SELECT SENSITIVE_key, val * 2 AS val FROM input",
      {CreateColumnSchema("SENSITIVE_key",
                          google::internal::federated::plan::STRING),
       CreateColumnSchema("val", google::internal::federated::plan::INT64)});
  configure_request.mutable_configure()->set_chunk_size(1000);
  configure_request.mutable_configure()->mutable_configuration()->PackFrom(
      query);
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_TRUE(stream->Read(&configure_response));

  SessionRequest write_request_1 = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE,
      BuildSensitiveGroupByCheckpoint({"k1", "k1", "k2"}, {1, 2, 5}));
  SessionResponse write_response_1;

  ASSERT_TRUE(stream->Write(write_request_1));
  EXPECT_TRUE(stream->Read(&write_response_1));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&finalize_response));

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("SENSITIVE_key_out");
  const absl::Span<const absl::string_view> output_keys =
      col_values->AsSpan<absl::string_view>();
  // The sensitive column has been hashed.
  EXPECT_THAT(output_keys, Not(AnyOf(Contains("k1"), Contains("k2"))));
}

TEST_F(FedSqlServerTest,
       ReportEncryptedInputsWithOutputNodeIdOutputsEncryptedResult) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  FedSqlContainerInitializeConfiguration init_config;
  init_config.set_report_output_access_policy_node_id(kReportOutputNodeId);
  *init_config.mutable_agg_configuration() = DefaultConfiguration();
  request.mutable_configuration()->PackFrom(init_config);
  request.set_max_num_sessions(kMaxNumSessions);

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      IsOk());

  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->set_chunk_size(1000);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_TRUE(stream->Read(&configure_response));
  auto nonce_generator =
      std::make_unique<NonceGenerator>(configure_response.configure().nonce());

  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_THAT(reencryption_public_key, IsOk());
  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();

  std::string message_0 = BuildFedSqlGroupByCheckpoint({9}, {1});
  absl::StatusOr<NonceAndCounter> nonce_0 = nonce_generator->GetNextBlobNonce();
  ASSERT_THAT(nonce_0, IsOk());
  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob_0 =
      crypto_test_utils::CreateRewrappedBlob(
          message_0, ciphertext_associated_data, response.public_key(),
          nonce_0->blob_nonce, *reencryption_public_key);
  ASSERT_THAT(rewrapped_blob_0, IsOk());

  SessionRequest request_0;
  WriteRequest* write_request_0 = request_0.mutable_write();
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  *write_request_0->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_0);
  write_request_0->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_0->counter);
  write_request_0->mutable_first_request_configuration()->PackFrom(config);
  write_request_0->set_commit(true);
  write_request_0->set_data(std::get<1>(*rewrapped_blob_0));

  SessionResponse response_0;

  ASSERT_TRUE(stream->Write(request_0));
  ASSERT_TRUE(stream->Read(&response_0));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream->Write(finalize_request));
  EXPECT_TRUE(stream->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(finalize_response.read()
                  .first_response_metadata()
                  .has_hpke_plus_aead_data());

  BlobMetadata::HpkePlusAeadMetadata result_metadata =
      finalize_response.read().first_response_metadata().hpke_plus_aead_data();

  BlobHeader result_header;
  EXPECT_TRUE(result_header.ParseFromString(
      result_metadata.ciphertext_associated_data()));
  EXPECT_EQ(result_header.access_policy_node_id(), kReportOutputNodeId);
  absl::StatusOr<std::string> decrypted_result =
      decryptor.Decrypt(finalize_response.read().data(),
                        result_metadata.ciphertext_associated_data(),
                        result_metadata.encrypted_symmetric_key(),
                        result_metadata.ciphertext_associated_data(),
                        result_metadata.encapsulated_public_key());
  ASSERT_THAT(decrypted_result, IsOk());
}

class InitializedFedSqlServerTest : public FedSqlServerTest {
 public:
  InitializedFedSqlServerTest() : FedSqlServerTest() {
    grpc::ClientContext context;
    InitializeRequest request;
    InitializeResponse response;
    FedSqlContainerInitializeConfiguration init_config;
    *init_config.mutable_agg_configuration() = DefaultConfiguration();
    request.mutable_configuration()->PackFrom(init_config);
    request.set_max_num_sessions(kMaxNumSessions);

    EXPECT_THAT(
        WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                               std::move(request)),
        IsOk());
    public_key_ = response.public_key();
  }

 protected:
  std::string public_key_;
};

TEST_F(InitializedFedSqlServerTest, InvalidConfigureRequest) {
  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  SqlQuery query;
  session_request.mutable_configure()->set_chunk_size(1000);
  session_request.mutable_configure()->mutable_configuration()->PackFrom(query);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_FALSE(stream->Read(&session_response));

  EXPECT_THAT(FromGrpcStatus(stream->Finish()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not contain exactly one table schema")));
}

TEST_F(InitializedFedSqlServerTest, ConfigureRequestWrongMessageType) {
  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  google::protobuf::Value value;
  session_request.mutable_configure()->set_chunk_size(1000);
  session_request.mutable_configure()->mutable_configuration()->PackFrom(value);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_FALSE(stream->Read(&session_response));

  EXPECT_THAT(FromGrpcStatus(stream->Finish()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("configuration cannot be unpacked")));
}

TEST_F(InitializedFedSqlServerTest, ConfigureInvalidTableSchema) {
  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  SqlQuery query;
  DatabaseSchema* input_schema = query.mutable_database_schema();
  input_schema->add_table();
  session_request.mutable_configure()->set_chunk_size(1000);
  session_request.mutable_configure()->mutable_configuration()->PackFrom(query);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_FALSE(stream->Read(&session_response));

  EXPECT_THAT(FromGrpcStatus(stream->Finish()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("SQL query input schema has no columns")));
}

TEST_F(InitializedFedSqlServerTest, SessionSqlQueryConfigureGeneratesNonce) {
  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_configure()->set_chunk_size(1000);
  session_request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSqlQuery());

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_TRUE(stream->Read(&session_response));

  ASSERT_TRUE(session_response.has_configure());
  ASSERT_GT(session_response.configure().nonce().size(), 0);
}

TEST_F(InitializedFedSqlServerTest, SessionEmptyConfigureGeneratesNonce) {
  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_configure()->set_chunk_size(1000);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_TRUE(stream->Read(&session_response));

  ASSERT_TRUE(session_response.has_configure());
  ASSERT_GT(session_response.configure().nonce().size(), 0);
}

TEST_F(InitializedFedSqlServerTest, SessionRejectsMoreThanMaximumNumSessions) {
  std::vector<std::unique_ptr<
      ::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>>
      streams;
  std::vector<std::unique_ptr<grpc::ClientContext>> contexts;
  for (int i = 0; i < kMaxNumSessions; i++) {
    std::unique_ptr<grpc::ClientContext> session_context =
        std::make_unique<grpc::ClientContext>();
    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure()->set_chunk_size(1000);
    session_request.mutable_configure()->mutable_configuration()->PackFrom(
        DefaultSqlQuery());

    std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
        stream = stub_->Session(session_context.get());
    ASSERT_TRUE(stream->Write(session_request));
    ASSERT_TRUE(stream->Read(&session_response));

    // Keep the context and stream so they don't go out of scope and end the
    // session.
    contexts.emplace_back(std::move(session_context));
    streams.emplace_back(std::move(stream));
  }

  grpc::ClientContext rejected_context;
  SessionRequest rejected_request;
  SessionResponse rejected_response;
  rejected_request.mutable_configure()->set_chunk_size(1000);
  rejected_request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSqlQuery());

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&rejected_context);
  ASSERT_TRUE(stream->Write(rejected_request));
  ASSERT_FALSE(stream->Read(&rejected_response));
  EXPECT_THAT(FromGrpcStatus(stream->Finish()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(InitializedFedSqlServerTest, SessionFailsIfSqlResultCannotBeAggregated) {
  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;

  TableSchema schema = CreateTableSchema(
      "input", "CREATE TABLE input (key INTEGER, val INTEGER)",
      {CreateColumnSchema("key", google::internal::federated::plan::INT64),
       CreateColumnSchema("val", google::internal::federated::plan::INT64)});
  SqlQuery query = CreateSqlQuery(
      schema, "SELECT key, val * 2 AS val_double FROM input",
      {CreateColumnSchema("key", google::internal::federated::plan::INT64),
       CreateColumnSchema("val_double",
                          google::internal::federated::plan::INT64)});

  // The output columns of the SQL query don't match the aggregation config, so
  // the results can't be aggregated.
  configure_request.mutable_configure()->set_chunk_size(1000);
  configure_request.mutable_configure()->mutable_configuration()->PackFrom(
      query);
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_TRUE(stream->Read(&configure_response));

  SessionRequest write_request_1 =
      CreateDefaultWriteRequest(AGGREGATION_TYPE_ACCUMULATE,
                                BuildFedSqlGroupByCheckpoint({7, 9}, {10, 12}));
  SessionResponse write_response_1;

  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_FALSE(stream->Read(&write_response_1));
  EXPECT_THAT(FromGrpcStatus(stream->Finish()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Failed to accumulate SQL query results")));
}

TEST_F(InitializedFedSqlServerTest, SessionWithoutSqlQuerySucceeds) {
  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  // No SQL query is configured.
  configure_request.mutable_configure()->set_chunk_size(1000);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_TRUE(stream->Read(&configure_response));

  SessionRequest write_request_1 = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({9}, {10}));
  SessionResponse write_response_1;

  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_TRUE(stream->Read(&write_response_1));
  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_TRUE(stream->Read(&write_response_1));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The aggregation sums the val column, grouping by the key column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 20);
}

TEST_F(InitializedFedSqlServerTest,
       SerializeEncryptedInputsWithoutOutputNodeIdFails) {
  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->set_chunk_size(1000);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_TRUE(stream->Read(&configure_response));
  auto nonce_generator =
      std::make_unique<NonceGenerator>(configure_response.configure().nonce());

  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_THAT(reencryption_public_key, IsOk());
  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();

  std::string message_0 = BuildFedSqlGroupByCheckpoint({8}, {1});
  absl::StatusOr<NonceAndCounter> nonce_0 = nonce_generator->GetNextBlobNonce();
  ASSERT_THAT(nonce_0, IsOk());
  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob_0 =
      crypto_test_utils::CreateRewrappedBlob(
          message_0, ciphertext_associated_data, public_key_,
          nonce_0->blob_nonce, *reencryption_public_key);
  ASSERT_THAT(rewrapped_blob_0, IsOk());

  SessionRequest request_0;
  WriteRequest* write_request_0 = request_0.mutable_write();
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  *write_request_0->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_0);
  write_request_0->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_0->counter);
  write_request_0->mutable_first_request_configuration()->PackFrom(config);
  write_request_0->set_commit(true);
  write_request_0->set_data(std::get<1>(*rewrapped_blob_0));

  SessionResponse response_0;

  ASSERT_TRUE(stream->Write(request_0));
  ASSERT_TRUE(stream->Read(&response_0));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_FALSE(stream->Read(&finalize_response));
  EXPECT_THAT(
      FromGrpcStatus(stream->Finish()),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "No output access policy node ID set for serialized outputs")));
}

class InitializedFedSqlServerKmsTest : public FedSqlServerTest {
 public:
  InitializedFedSqlServerKmsTest() : FedSqlServerTest() {
    grpc::ClientContext context;
    InitializeRequest request;
    InitializeResponse response;
    FedSqlContainerInitializeConfiguration init_config;
    *init_config.mutable_agg_configuration() = DefaultConfiguration();
    request.mutable_configuration()->PackFrom(init_config);
    request.set_max_num_sessions(kMaxNumSessions);

    auto public_private_key_pair = crypto_test_utils::GenerateKeyPair(key_id_);
    public_key_ = public_private_key_pair.first;

    FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
      intrinsic_uri: "fedsql_group_by"
      access_budget { times: 5 }
    )pb");
    AuthorizeConfidentialTransformResponse::ProtectedResponse
        protected_response;
    // Add 2 re-encryption keys - Merge and Report.
    protected_response.add_result_encryption_keys(public_key_);
    protected_response.add_result_encryption_keys(public_key_);
    protected_response.add_decryption_keys(public_private_key_pair.second);
    AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
    associated_data.mutable_config_constraints()->PackFrom(config_constraints);
    associated_data.add_authorized_logical_pipeline_policies_hashes(
        allowed_policy_hash_);
    auto encrypted_request =
        oak_client_encryptor_
            ->Encrypt(protected_response.SerializeAsString(),
                      associated_data.SerializeAsString())
            .value();
    *request.mutable_protected_response() = encrypted_request;

    auto writer = stub_->StreamInitialize(&context, &response);
    EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
    EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
                IsOk());

    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure()->set_chunk_size(1000);
    session_request.mutable_configure()->mutable_configuration()->PackFrom(
        DefaultSqlQuery());

    stream_ = stub_->Session(&session_context_);
    CHECK(stream_->Write(session_request));
    CHECK(stream_->Read(&session_response));
  }

  std::pair<BlobMetadata, std::string> EncryptWithKmsKeys(
      std::string message, std::string associated_data) {
    MessageEncryptor encryptor;
    absl::StatusOr<EncryptMessageResult> encrypt_result =
        encryptor.Encrypt(message, public_key_, associated_data);
    CHECK(encrypt_result.ok()) << encrypt_result.status();

    BlobMetadata metadata;
    metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
    metadata.set_total_size_bytes(encrypt_result.value().ciphertext.size());
    BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
        metadata.mutable_hpke_plus_aead_data();
    encryption_metadata->set_ciphertext_associated_data(associated_data);
    encryption_metadata->set_encrypted_symmetric_key(
        encrypt_result.value().encrypted_symmetric_key);
    encryption_metadata->set_encapsulated_public_key(
        encrypt_result.value().encapped_key);
    encryption_metadata->mutable_kms_symmetric_key_associated_data()
        ->set_record_header(associated_data);

    return {metadata, encrypt_result.value().ciphertext};
  }

 protected:
  grpc::ClientContext session_context_;
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream_;
  std::string key_id_ = "key_id";
  std::string allowed_policy_hash_ = "hash_1";
  std::string public_key_;
};

TEST_F(InitializedFedSqlServerKmsTest, SessionWriteRequestSuccess) {
  std::string message = BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0});
  BlobHeader header;
  header.set_blob_id("blob_id");
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  auto [metadata, ciphertext] =
      EncryptWithKmsKeys(message, header.SerializeAsString());

  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  write_request->mutable_first_request_configuration()->PackFrom(config);
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->set_commit(true);
  write_request->set_data(ciphertext);

  SessionResponse response;
  ASSERT_TRUE(stream_->Write(request));
  ASSERT_TRUE(stream_->Read(&response));
  ASSERT_EQ(response.write().status().code(), Code::OK)
      << response.write().status().message();
  ASSERT_EQ(response.write().committed_size_bytes(), ciphertext.size());
}

TEST_F(InitializedFedSqlServerKmsTest, SessionWriteRequestNoKmsAssociatedData) {
  std::string message = BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0});
  BlobHeader header;
  header.set_blob_id("blob_id");
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  auto [metadata, ciphertext] =
      EncryptWithKmsKeys(message, header.SerializeAsString());
  metadata.mutable_hpke_plus_aead_data()
      ->clear_kms_symmetric_key_associated_data();

  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  write_request->mutable_first_request_configuration()->PackFrom(config);
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->set_commit(true);
  write_request->set_data(ciphertext);

  SessionResponse response;
  ASSERT_TRUE(stream_->Write(request));
  ASSERT_TRUE(stream_->Read(&response));
  ASSERT_EQ(response.write().status().code(), Code::INVALID_ARGUMENT)
      << response.write().status().message();
}

TEST_F(InitializedFedSqlServerKmsTest,
       SessionWriteRequestInvalidKmsAssociatedData) {
  std::string message = BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0});
  auto [metadata, ciphertext] =
      EncryptWithKmsKeys(message, "invalid_associated_data");

  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  write_request->mutable_first_request_configuration()->PackFrom(config);
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->set_commit(true);
  write_request->set_data(ciphertext);

  SessionResponse response;
  ASSERT_TRUE(stream_->Write(request));
  ASSERT_TRUE(stream_->Read(&response));
  ASSERT_EQ(response.write().status().code(), Code::INVALID_ARGUMENT)
      << response.write().status().message();
}

TEST_F(InitializedFedSqlServerKmsTest, SessionWriteRequestInvalidPolicyHash) {
  std::string message = BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0});
  BlobHeader header;
  header.set_blob_id("blob_id");
  header.set_key_id(key_id_);
  header.set_access_policy_sha256("invalid_policy_hash");
  auto [metadata, ciphertext] =
      EncryptWithKmsKeys(message, header.SerializeAsString());

  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  write_request->mutable_first_request_configuration()->PackFrom(config);
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->set_commit(true);
  write_request->set_data(ciphertext);

  SessionResponse response;
  ASSERT_TRUE(stream_->Write(request));
  ASSERT_TRUE(stream_->Read(&response));
  ASSERT_EQ(response.write().status().code(), Code::INVALID_ARGUMENT)
      << response.write().status().message();
}

class FedSqlGroupByTest : public FedSqlServerTest {
 public:
  FedSqlGroupByTest() {
    grpc::ClientContext context;
    InitializeRequest request;
    InitializeResponse response;
    FedSqlContainerInitializeConfiguration init_config;
    init_config.set_serialize_output_access_policy_node_id(
        kSerializeOutputNodeId);
    *init_config.mutable_agg_configuration() = DefaultConfiguration();
    request.mutable_configuration()->PackFrom(init_config);
    request.set_max_num_sessions(kMaxNumSessions);

    EXPECT_THAT(
        WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                               std::move(request)),
        IsOk());
    public_key_ = response.public_key();

    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure()->set_chunk_size(1000);
    session_request.mutable_configure()->mutable_configuration()->PackFrom(
        DefaultSqlQuery());

    stream_ = stub_->Session(&session_context_);
    CHECK(stream_->Write(session_request));
    CHECK(stream_->Read(&session_response));
    nonce_generator_ =
        std::make_unique<NonceGenerator>(session_response.configure().nonce());
  }

 protected:
  grpc::ClientContext session_context_;
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream_;
  std::unique_ptr<NonceGenerator> nonce_generator_;
  std::string public_key_;
};

TEST_F(FedSqlGroupByTest, SessionExecutesSqlQueryAndAggregation) {
  SessionRequest write_request_1 = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE,
      BuildFedSqlGroupByCheckpoint({1, 1, 2}, {1, 2, 5}));
  SessionResponse write_response_1;

  ASSERT_TRUE(stream_->Write(write_request_1));
  ASSERT_TRUE(stream_->Read(&write_response_1));

  SessionRequest write_request_2 =
      CreateDefaultWriteRequest(AGGREGATION_TYPE_ACCUMULATE,
                                BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0}));
  SessionResponse write_response_2;
  ASSERT_TRUE(stream_->Write(write_request_2));
  ASSERT_TRUE(stream_->Read(&write_response_2));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The SQL query doubles each `val`, and the aggregation sums the val
  // column, grouping by key.
  ASSERT_EQ(col_values->num_elements(), 3);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  EXPECT_THAT(col_values->AsSpan<int64_t>(), UnorderedElementsAre(14, 10, 0));
}

TEST_F(FedSqlGroupByTest, SessionWriteAccumulateCommitsBlob) {
  FederatedComputeCheckpointParserFactory parser_factory;

  std::string data = BuildFedSqlGroupByCheckpoint({8}, {1});
  SessionRequest write_request =
      CreateDefaultWriteRequest(AGGREGATION_TYPE_ACCUMULATE, data);
  SessionResponse write_response;

  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  ASSERT_TRUE(write_response.has_write());
  ASSERT_EQ(write_response.write().committed_size_bytes(), data.size());
  ASSERT_EQ(write_response.write().status().code(), Code::OK)
      << write_response.write().status().message();
}

TEST_F(FedSqlGroupByTest, SessionAccumulatesAndReports) {
  SessionRequest write_request = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({7}, {1}));
  SessionResponse write_response;

  // Accumulate the same unencrypted blob twice.
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The SQL query doubles each input and the aggregation sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 4);
}

TEST_F(FedSqlGroupByTest, SessionAccumulatesAndSerializes) {
  SessionRequest write_request = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({9}, {1}));
  SessionResponse write_response;

  // Accumulate the same unencrypted blob twice.
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  std::string data = finalize_response.read().data();
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), data);
  ASSERT_THAT(deserialized_agg, IsOk());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_THAT((*deserialized_agg)->Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  auto col_values = (*parser)->GetTensor("val_out");
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 4);
}

TEST_F(FedSqlGroupByTest, SessionMergesAndReports) {
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({4}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_THAT(input_aggregator->Accumulate(*input_parser), IsOk());

  SessionRequest write_request = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_MERGE, std::move(*input_aggregator).Serialize().value());
  SessionResponse write_response;

  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The input aggregator should be merged with the session aggregator
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 3);
}

TEST_F(FedSqlGroupByTest, SerializeZeroInputsProducesEmptyOutput) {
  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());
  std::string data = finalize_response.read().data();
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>>
      deserialized_agg_status =
          CheckpointAggregator::Deserialize(DefaultConfiguration(), data);
  ASSERT_THAT(deserialized_agg_status, IsOk());
  std::unique_ptr<CheckpointAggregator> deserialized_agg =
      *std::move(deserialized_agg_status);

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();

  EXPECT_THAT(deserialized_agg->GetNumCheckpointsAggregated(), IsOkAndHolds(0));

  // Merging the empty deserialized aggregator with another aggregator should
  // have no effect on the output of the other aggregator.
  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({2}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> other_aggregator =
      CheckpointAggregator::Create(DefaultConfiguration()).value();
  ASSERT_THAT(other_aggregator->Accumulate(*input_parser), IsOk());

  ASSERT_THAT(other_aggregator->MergeWith(std::move(*deserialized_agg)),
              IsOk());

  ASSERT_THAT((*other_aggregator).Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  ASSERT_THAT(checkpoint, IsOk());
  auto parser = parser_factory.Create(*checkpoint);
  auto col_values = (*parser)->GetTensor("val_out");
  // The value from other_aggregator is unchanged by deserialized_agg
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 3);
}

TEST_F(FedSqlGroupByTest, ReportZeroInputsReturnsInvalidArgument) {
  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_FALSE(stream_->Read(&finalize_response));
  EXPECT_THAT(FromGrpcStatus(stream_->Finish()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(FedSqlGroupByTest, SessionIgnoresUnparseableInputs) {
  SessionRequest write_request_1 = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({8}, {7}));
  SessionResponse write_response_1;

  ASSERT_TRUE(stream_->Write(write_request_1));
  ASSERT_TRUE(stream_->Read(&write_response_1));

  SessionRequest invalid_write = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, "invalid checkpoint");
  SessionResponse invalid_write_response;

  ASSERT_TRUE(stream_->Write(invalid_write));
  ASSERT_TRUE(stream_->Read(&invalid_write_response));

  ASSERT_TRUE(invalid_write_response.has_write());
  ASSERT_EQ(invalid_write_response.write().committed_size_bytes(), 0);
  ASSERT_EQ(invalid_write_response.write().status().code(),
            Code::INVALID_ARGUMENT)
      << invalid_write_response.write().status().message();

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The invalid input should be ignored
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 14);
}

TEST_F(FedSqlGroupByTest, SessionIgnoresInputThatCannotBeQueried) {
  SessionRequest write_request_1 = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE,
      BuildFedSqlGroupByCheckpoint({9}, {7}, "bad_key_col_name"));
  SessionResponse write_response_1;

  ASSERT_TRUE(stream_->Write(write_request_1));
  ASSERT_TRUE(stream_->Read(&write_response_1));
  ASSERT_EQ(write_response_1.write().status().code(), Code::NOT_FOUND)
      << write_response_1.write().status().message();
}

TEST_F(FedSqlGroupByTest, SessionDecryptsMultipleRecordsAndReports) {
  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_THAT(reencryption_public_key, IsOk());
  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();

  std::string message_0 = BuildFedSqlGroupByCheckpoint({9}, {1});
  absl::StatusOr<NonceAndCounter> nonce_0 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_THAT(nonce_0, IsOk());
  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob_0 =
      crypto_test_utils::CreateRewrappedBlob(
          message_0, ciphertext_associated_data, public_key_,
          nonce_0->blob_nonce, *reencryption_public_key);
  ASSERT_THAT(rewrapped_blob_0, IsOk());

  SessionRequest request_0;
  WriteRequest* write_request_0 = request_0.mutable_write();
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  *write_request_0->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_0);
  write_request_0->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_0->counter);
  write_request_0->mutable_first_request_configuration()->PackFrom(config);
  write_request_0->set_commit(true);
  write_request_0->set_data(std::get<1>(*rewrapped_blob_0));

  SessionResponse response_0;

  ASSERT_TRUE(stream_->Write(request_0));
  ASSERT_TRUE(stream_->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), Code::OK)
      << response_0.write().status().message();

  std::string message_1 = BuildFedSqlGroupByCheckpoint({9}, {2});
  absl::StatusOr<NonceAndCounter> nonce_1 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_THAT(nonce_1, IsOk());
  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob_1 =
      crypto_test_utils::CreateRewrappedBlob(
          message_1, ciphertext_associated_data, public_key_,
          nonce_1->blob_nonce, *reencryption_public_key);
  ASSERT_THAT(rewrapped_blob_1, IsOk());

  SessionRequest request_1;
  WriteRequest* write_request_1 = request_1.mutable_write();
  *write_request_1->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_1);
  write_request_1->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_1->counter);
  write_request_1->mutable_first_request_configuration()->PackFrom(config);
  write_request_1->set_commit(true);
  write_request_1->set_data(std::get<1>(*rewrapped_blob_1));

  SessionResponse response_1;

  ASSERT_TRUE(stream_->Write(request_1));
  ASSERT_TRUE(stream_->Read(&response_1));
  ASSERT_EQ(response_1.write().status().code(), Code::OK)
      << response_1.write().status().message();

  std::string message_2 = BuildFedSqlGroupByCheckpoint({9}, {3});
  absl::StatusOr<NonceAndCounter> nonce_2 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_THAT(nonce_2, IsOk());
  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob_2 =
      crypto_test_utils::CreateRewrappedBlob(
          message_2, ciphertext_associated_data, public_key_,
          nonce_2->blob_nonce, *reencryption_public_key);
  ASSERT_THAT(rewrapped_blob_2, IsOk());

  SessionRequest request_2;
  WriteRequest* write_request_2 = request_2.mutable_write();
  *write_request_2->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_2);
  write_request_2->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_2->counter);
  write_request_2->mutable_first_request_configuration()->PackFrom(config);
  write_request_2->set_commit(true);
  write_request_2->set_data(std::get<1>(*rewrapped_blob_2));

  SessionResponse response_2;

  ASSERT_TRUE(stream_->Write(request_2));
  ASSERT_TRUE(stream_->Read(&response_2));
  ASSERT_EQ(response_2.write().status().code(), Code::OK)
      << response_2.write().status().message();

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The SQL query doubles every input and the aggregation sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 12);
}

TEST_F(FedSqlGroupByTest, SessionDecryptsMultipleRecordsAndSerializes) {
  MessageDecryptor earliest_decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      earliest_decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_THAT(reencryption_public_key, IsOk());

  absl::StatusOr<OkpCwt> reencryption_okp_cwt =
      OkpCwt::Decode(*reencryption_public_key);
  ASSERT_THAT(reencryption_okp_cwt, IsOk());
  reencryption_okp_cwt->expiration_time = absl::FromUnixSeconds(1);
  absl::StatusOr<std::string> earliest_reencryption_key =
      reencryption_okp_cwt->Encode();
  ASSERT_THAT(earliest_reencryption_key, IsOk());
  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();

  std::string message_0 = BuildFedSqlGroupByCheckpoint({7}, {1});
  absl::StatusOr<NonceAndCounter> nonce_0 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_THAT(nonce_0, IsOk());
  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob_0 =
      crypto_test_utils::CreateRewrappedBlob(
          message_0, ciphertext_associated_data, public_key_,
          nonce_0->blob_nonce, *earliest_reencryption_key);
  ASSERT_THAT(rewrapped_blob_0, IsOk());

  SessionRequest request_0;
  WriteRequest* write_request_0 = request_0.mutable_write();
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  *write_request_0->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_0);
  write_request_0->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_0->counter);
  write_request_0->mutable_first_request_configuration()->PackFrom(config);
  write_request_0->set_commit(true);
  write_request_0->set_data(std::get<1>(*rewrapped_blob_0));

  SessionResponse response_0;

  ASSERT_TRUE(stream_->Write(request_0));
  ASSERT_TRUE(stream_->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), Code::OK)
      << response_0.write().status().message();

  std::string message_1 = BuildFedSqlGroupByCheckpoint({7}, {2});
  absl::StatusOr<NonceAndCounter> nonce_1 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_THAT(nonce_1, IsOk());

  MessageDecryptor later_decryptor;
  absl::StatusOr<std::string> other_reencryption_public_key =
      later_decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_THAT(other_reencryption_public_key, IsOk());

  absl::StatusOr<OkpCwt> other_reencryption_okp_cwt =
      OkpCwt::Decode(*other_reencryption_public_key);
  ASSERT_THAT(other_reencryption_okp_cwt, IsOk());
  other_reencryption_okp_cwt->expiration_time = absl::FromUnixSeconds(99);
  absl::StatusOr<std::string> later_reencryption_key =
      reencryption_okp_cwt->Encode();
  ASSERT_THAT(later_reencryption_key, IsOk());
  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob_1 =
      crypto_test_utils::CreateRewrappedBlob(
          message_1, ciphertext_associated_data, public_key_,
          nonce_1->blob_nonce, *later_reencryption_key);
  ASSERT_THAT(rewrapped_blob_1, IsOk());

  SessionRequest request_1;
  WriteRequest* write_request_1 = request_1.mutable_write();
  *write_request_1->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_1);
  write_request_1->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_1->counter);
  write_request_1->mutable_first_request_configuration()->PackFrom(config);
  write_request_1->set_commit(true);
  write_request_1->set_data(std::get<1>(*rewrapped_blob_1));

  SessionResponse response_1;

  ASSERT_TRUE(stream_->Write(request_1));
  ASSERT_TRUE(stream_->Read(&response_1));
  ASSERT_EQ(response_1.write().status().code(), Code::OK)
      << response_1.write().status().message();

  // Unencrypted request should be incorporated, but the serialized result
  // should still be encrypted.
  SessionRequest unencrypted_request = CreateDefaultWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({7}, {3}));
  SessionResponse unencrypted_response;

  ASSERT_TRUE(stream_->Write(unencrypted_request));
  ASSERT_TRUE(stream_->Read(&unencrypted_response));
  ASSERT_EQ(unencrypted_response.write().status().code(), Code::OK)
      << unencrypted_response.write().status().message();

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  EXPECT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(finalize_response.read()
                  .first_response_metadata()
                  .has_hpke_plus_aead_data());

  BlobMetadata::HpkePlusAeadMetadata result_metadata =
      finalize_response.read().first_response_metadata().hpke_plus_aead_data();

  BlobHeader result_header;
  EXPECT_TRUE(result_header.ParseFromString(
      result_metadata.ciphertext_associated_data()));
  EXPECT_EQ(result_header.access_policy_node_id(), kSerializeOutputNodeId);
  // The decryptor with the earliest set expiration time should be able to
  // decrypt the encrypted results. The later decryptor should not.
  absl::StatusOr<std::string> decrypted_result =
      earliest_decryptor.Decrypt(finalize_response.read().data(),
                                 result_metadata.ciphertext_associated_data(),
                                 result_metadata.encrypted_symmetric_key(),
                                 result_metadata.ciphertext_associated_data(),
                                 result_metadata.encapsulated_public_key());
  ASSERT_THAT(decrypted_result, IsOk());
  absl::StatusOr<std::string> failed_decrypt =
      later_decryptor.Decrypt(finalize_response.read().data(),
                              result_metadata.ciphertext_associated_data(),
                              result_metadata.encrypted_symmetric_key(),
                              result_metadata.ciphertext_associated_data(),
                              result_metadata.encapsulated_public_key());
  ASSERT_FALSE(failed_decrypt.ok()) << failed_decrypt.status();

  std::string decrypted_data = *decrypted_result;
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(DefaultConfiguration(), decrypted_data);
  ASSERT_THAT(deserialized_agg, IsOk());

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  ASSERT_THAT((*deserialized_agg)->Report(*checkpoint_builder), IsOk());
  absl::StatusOr<absl::Cord> checkpoint = checkpoint_builder->Build();
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(*checkpoint);
  auto col_values = (*parser)->GetTensor("val_out");
  // The query sums the input column
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 12);
}

TEST_F(FedSqlGroupByTest, SessionIgnoresUndecryptableInputs) {
  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_THAT(reencryption_public_key, IsOk());
  std::string ciphertext_associated_data = "ciphertext associated data";

  // Create one blob that will fail to decrypt and one blob that can be
  // successfully decrypted.
  std::string message_0 = BuildFedSqlGroupByCheckpoint({42}, {1});
  absl::StatusOr<NonceAndCounter> nonce_0 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_THAT(nonce_0, IsOk());
  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob_0 =
      crypto_test_utils::CreateRewrappedBlob(
          message_0, ciphertext_associated_data, public_key_,
          nonce_0->blob_nonce, *reencryption_public_key);
  ASSERT_THAT(rewrapped_blob_0, IsOk());

  SessionRequest request_0;
  WriteRequest* write_request_0 = request_0.mutable_write();
  FedSqlContainerWriteConfiguration config = PARSE_TEXT_PROTO(R"pb(
    type: AGGREGATION_TYPE_ACCUMULATE
  )pb");
  *write_request_0->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_0);
  write_request_0->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_0->counter);
  write_request_0->mutable_first_request_configuration()->PackFrom(config);
  write_request_0->set_commit(true);
  write_request_0->set_data("undecryptable");

  SessionResponse response_0;

  ASSERT_TRUE(stream_->Write(request_0));
  ASSERT_TRUE(stream_->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), Code::INVALID_ARGUMENT)
      << response_0.write().status().message();

  std::string message_1 = BuildFedSqlGroupByCheckpoint({42}, {2});
  absl::StatusOr<NonceAndCounter> nonce_1 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_THAT(nonce_1, IsOk());
  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob_1 =
      crypto_test_utils::CreateRewrappedBlob(
          message_1, ciphertext_associated_data, public_key_,
          nonce_1->blob_nonce, *reencryption_public_key);
  ASSERT_THAT(rewrapped_blob_1, IsOk());

  SessionRequest request_1;
  WriteRequest* write_request_1 = request_1.mutable_write();
  *write_request_1->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_1);
  write_request_1->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_1->counter);
  write_request_1->mutable_first_request_configuration()->PackFrom(config);
  write_request_1->set_commit(true);
  write_request_1->set_data(std::get<1>(*rewrapped_blob_1));

  SessionResponse response_1;

  ASSERT_TRUE(stream_->Write(request_1));
  ASSERT_TRUE(stream_->Read(&response_1));
  ASSERT_EQ(response_1.write().status().code(), Code::OK)
      << response_1.write().status().message();

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(finalize_response.read().data());
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The undecryptable write is ignored, and only the valid write is aggregated.
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 4);
}

TEST_F(FedSqlServerTest,
       StreamInitializeWithGemmaInferenceSessionMissingTokenizerId) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest initialize_request;
  FedSqlContainerInitializeConfiguration init_config = PARSE_TEXT_PROTO(R"pb(
    agg_configuration {
      intrinsic_configs: {
        intrinsic_uri: "fedsql_group_by"
        intrinsic_args {
          input_tensor {
            name: "topic"
            dtype: DT_STRING
            shape { dim_sizes: -1 }
          }
        }
        output_tensors {
          name: "topic_agg"
          dtype: DT_STRING
          shape { dim_sizes: -1 }
        }
        inner_intrinsics {
          intrinsic_uri: "GoogleSQL:sum"
          intrinsic_args {
            input_tensor {
              name: "topic_count"
              dtype: DT_INT64
              shape {}
            }
          }
          output_tensors {
            name: "topic_count_agg"
            dtype: DT_INT64
            shape {}
          }
        }
      }
    }
    serialize_output_access_policy_node_id: 42
    report_output_access_policy_node_id: 7
    inference_init_config {
      inference_config {
        inference_task: {
          column_config {
            input_column_name: "transcript"
            output_column_name: "topic"
          }
          prompt { prompt_template: "Hello, {{transcript}}" }
        }
        gemma_config {
          tokenizer_file: "/path/to/tokenizer"
          model_weight_file: "/path/to/model_weight"
          model: GEMMA_MODEL_UNSPECIFIED
          model_training: GEMMA_IT
          tensor_type: GEMMA_SFP
        }
      }
      gemma_init_config {
        model_weight_configuration_id: "gemma_model_weight_id"
      }
    }
  )pb");
  initialize_request.mutable_configuration()->PackFrom(init_config);
  initialize_request.set_max_num_sessions(kMaxNumSessions);

  // Set tokenizer data blob.
  std::string expected_tokenizer_content = "tokenizer content";
  StreamInitializeRequest tokenizer_write_config;
  ConfigurationMetadata* tokenizer_metadata =
      tokenizer_write_config.mutable_write_configuration()
          ->mutable_first_request_metadata();
  tokenizer_metadata->set_configuration_id("gemma_tokenizer_id");
  tokenizer_write_config.mutable_write_configuration()->set_commit(true);
  absl::Cord tokenizer_content(expected_tokenizer_content);
  tokenizer_metadata->set_total_size_bytes(tokenizer_content.size());
  std::string tokenizer_content_string;
  absl::CopyCordToString(tokenizer_content, &tokenizer_content_string);
  tokenizer_write_config.mutable_write_configuration()->set_data(
      tokenizer_content_string);

  // Set up model weight data blob.
  // Reuse data for the first and second WriteConfigurationRequest for the model
  // weight blob.
  std::string expected_model_weight_content = "first model weight content";
  absl::Cord model_weight_content(expected_model_weight_content);
  std::string model_weight_content_string;
  absl::CopyCordToString(model_weight_content, &model_weight_content_string);

  // Set the first WriteConfigurationRequest for the model weight data blob.
  StreamInitializeRequest first_model_weight_write_config;
  ConfigurationMetadata* first_model_weight_metadata =
      first_model_weight_write_config.mutable_write_configuration()
          ->mutable_first_request_metadata();
  first_model_weight_metadata->set_total_size_bytes(
      model_weight_content.size() * 2);
  first_model_weight_metadata->set_configuration_id("gemma_model_weight_id");

  first_model_weight_write_config.mutable_write_configuration()->set_data(
      model_weight_content_string);

  // Set the second WriteConfigurationRequest for the model weight data blob.
  StreamInitializeRequest second_model_weight_write_config;
  second_model_weight_write_config.mutable_write_configuration()->set_commit(
      true);
  second_model_weight_write_config.mutable_write_configuration()->set_data(
      model_weight_content_string);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(tokenizer_write_config));
  ASSERT_TRUE(writer->Write(first_model_weight_write_config));
  ASSERT_TRUE(writer->Write(second_model_weight_write_config));

  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Expected Gemma tokenizer configuration id  is missing "
                    "in WriteConfigurationRequest")));

  // Remove inference files after assertions.
  std::filesystem::remove("/tmp/write_configuration_1");
  std::filesystem::remove("/tmp/write_configuration_2");
}

}  // namespace

}  // namespace confidential_federated_compute::fed_sql
