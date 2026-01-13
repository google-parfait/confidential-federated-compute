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
#include "containers/big_endian.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "containers/fed_sql/budget.pb.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/fed_sql/testing/mocks.h"
#include "containers/fed_sql/testing/test_utils.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/private_state.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
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
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
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
using ::confidential_federated_compute::fed_sql::testing::
    BuildMessageCheckpoint;
using ::confidential_federated_compute::fed_sql::testing::MessageHelper;
using ::confidential_federated_compute::fed_sql::testing::MockInferenceModel;
using ::fcp::base::FromGrpcStatus;
using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivateStateConfigId;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::ReleaseToken;
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
using ::fcp::confidentialcompute::FedSqlContainerCommitConfiguration;
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
using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::DynamicMessageFactory;
using ::google::protobuf::FileDescriptorProto;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;
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
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::AnyOf;
using ::testing::ByMove;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Not;
using ::testing::Pair;
using ::testing::Return;
using ::testing::SizeIs;
using ::testing::Test;
using ::testing::UnorderedElementsAre;

inline constexpr int kMaxNumSessions = 8;

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

FedSqlContainerInitializeConfiguration DefaultFedSqlContainerConfig() {
  return PARSE_TEXT_PROTO(R"pb(
    agg_configuration {
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
    }
  )pb");
}

FedSqlContainerInitializeConfiguration DefaultFedSqlDpContainerConfig() {
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
  )pb");
}

FedSqlContainerConfigConstraints DefaultFedSqlConfigConstraints() {
  return PARSE_TEXT_PROTO(R"pb(
    intrinsic_uris: "fedsql_group_by"
    access_budget { times: 5 }
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

InferenceInitializeConfiguration DefaultInferenceConfig() {
  return PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
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
        weight_type: WEIGHT_TYPE_SBS
      }
    }
    gemma_init_config {
      tokenizer_configuration_id: "gemma_tokenizer_id"
      model_weight_configuration_id: "gemma_model_weight_id"
    }
  )pb");
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

    ON_CALL(*mock_inference_model_, BuildGemmaCppModel)
        .WillByDefault(Return(absl::OkStatus()));
    ON_CALL(*mock_inference_model_, RunGemmaCppInference)
        .WillByDefault([](const fcp::confidentialcompute::Prompt&,
                          const sql::Input&, absl::Span<const size_t>,
                          const std::string&)
                           -> absl::StatusOr<InferenceModel::InferenceOutput> {
          std::initializer_list<absl::string_view> topic_values = {
              "topic_value"};
          absl::StatusOr<Tensor> tensor = Tensor::Create(
              DataType::DT_STRING,
              TensorShape({static_cast<int64_t>(topic_values.size())}),
              std::make_unique<MutableVectorData<absl::string_view>>(
                  topic_values),
              /*name=*/"topic");
          if (!tensor.ok()) {
            return tensor.status();
          }
          return InferenceModel::InferenceOutput{.tensor = std::move(*tensor),
                                                 .per_row_output_counts = {1}};
        });
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
  InitializeRequest CreateInitializeRequest(
      FedSqlContainerInitializeConfiguration init_config,
      FedSqlContainerConfigConstraints config_constraints =
          DefaultFedSqlConfigConstraints()) {
    InitializeRequest request;
    request.mutable_configuration()->PackFrom(init_config);
    request.set_max_num_sessions(kMaxNumSessions);

    auto public_private_key_pair = crypto_test_utils::GenerateKeyPair(key_id_);
    public_key_ = public_private_key_pair.first;
    message_decryptor_ =
        std::make_unique<MessageDecryptor>(std::vector<absl::string_view>(
            {public_private_key_pair.second, public_private_key_pair.second}));

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
    associated_data.add_omitted_decryption_key_ids("foo");
    auto encrypted_request =
        oak_client_encryptor_
            ->Encrypt(protected_response.SerializeAsString(),
                      associated_data.SerializeAsString())
            .value();
    *request.mutable_protected_response() = encrypted_request;
    return request;
  }

  InitializeRequest CreateInitializeRequest() {
    return CreateInitializeRequest(DefaultFedSqlContainerConfig());
  }

  void InitializeTransform() {
    grpc::ClientContext context;
    InitializeRequest request = CreateInitializeRequest();
    InitializeResponse response;
    auto writer = stub_->StreamInitialize(&context, &response);
    BudgetState budget_state =
        PARSE_TEXT_PROTO(R"pb(buckets { key: "expired_key" budget: 3 }
                              buckets { key: "foo" budget: 3 })pb");
    EXPECT_TRUE(WritePipelinePrivateState(writer.get(),
                                          budget_state.SerializeAsString()));
    EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
                IsOk());
  }

  SessionRequest CreateDefaultEncryptedWriteRequest(
      AggCoreAggregationType agg_type, std::string data,
      std::string associated_data) {
    auto [metadata, ciphertext] = Encrypt(data, associated_data);

    SessionRequest request;
    WriteRequest* write_request = request.mutable_write();
    FedSqlContainerWriteConfiguration config;
    config.set_type(agg_type);
    write_request->mutable_first_request_configuration()->PackFrom(config);
    *write_request->mutable_first_request_metadata() = metadata;
    write_request->set_commit(true);
    write_request->set_data(ciphertext);
    return request;
  }

  std::pair<BlobMetadata, std::string> Encrypt(std::string message,
                                               std::string associated_data) {
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

  std::string Decrypt(BlobMetadata metadata, absl::string_view ciphertext) {
    BlobHeader blob_header;
    blob_header.ParseFromString(metadata.hpke_plus_aead_data()
                                    .kms_symmetric_key_associated_data()
                                    .record_header());
    auto decrypted = message_decryptor_->Decrypt(
        ciphertext, metadata.hpke_plus_aead_data().ciphertext_associated_data(),
        metadata.hpke_plus_aead_data().encrypted_symmetric_key(),
        metadata.hpke_plus_aead_data()
            .kms_symmetric_key_associated_data()
            .record_header(),
        metadata.hpke_plus_aead_data().encapsulated_public_key(),
        blob_header.key_id());
    CHECK_OK(decrypted.status());
    return decrypted.value();
  }

  std::unique_ptr<grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
  ConfigureDefaultSession(grpc::ClientContext* session_context) {
    SessionRequest configure_request;
    SessionResponse configure_response;
    configure_request.mutable_configure()->set_chunk_size(1000);
    configure_request.mutable_configure()->mutable_configuration()->PackFrom(
        DefaultSqlQuery());
    auto stream = stub_->Session(session_context);
    EXPECT_TRUE(stream->Write(configure_request));
    EXPECT_TRUE(stream->Read(&configure_response));
    return stream;
  }

  std::shared_ptr<NiceMock<MockInferenceModel>> mock_inference_model_ =
      std::make_shared<NiceMock<MockInferenceModel>>();
  std::unique_ptr<FedSqlConfidentialTransform> service_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
  std::unique_ptr<ClientEncryptor> oak_client_encryptor_;
  std::string key_id_ = "key_id";
  std::string allowed_policy_hash_ = "hash_1";
  std::string public_key_;
  std::unique_ptr<MessageDecryptor> message_decryptor_;
};

TEST_F(FedSqlServerTest, StreamInitializeWithMessageConfigSucceeds) {
  MessageHelper message_helper;
  google::protobuf::FileDescriptorSet descriptor_set;
  message_helper.file_descriptor()->CopyTo(descriptor_set.add_file());
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  init_config.mutable_private_logger_uploads_config()
      ->mutable_message_description()
      ->set_message_descriptor_set(descriptor_set.SerializeAsString());
  init_config.mutable_private_logger_uploads_config()
      ->mutable_message_description()
      ->set_message_name(message_helper.message_name());
  InitializeRequest request = CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());
}

TEST_F(FedSqlServerTest, StreamInitializeMissingMessageNameFails) {
  MessageHelper message_helper;
  google::protobuf::FileDescriptorSet descriptor_set;
  message_helper.file_descriptor()->CopyTo(descriptor_set.add_file());
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  init_config.mutable_private_logger_uploads_config()
      ->mutable_message_description()
      ->set_message_descriptor_set(descriptor_set.SerializeAsString());
  InitializeRequest request = CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("If private_logger_uploads_config is set, "
                                 "both message_descriptor_set and message_name "
                                 "must be set within message_description")));
}

TEST_F(FedSqlServerTest, StreamInitializeMissingMessageDescriptorSetFails) {
  const google::protobuf::Descriptor* descriptor =
      fcp::confidentialcompute::InitializeRequest::descriptor();
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  init_config.mutable_private_logger_uploads_config()
      ->mutable_message_description()
      ->set_message_name(descriptor->full_name());
  InitializeRequest request = CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("If private_logger_uploads_config is set, "
                                 "both message_descriptor_set and message_name "
                                 "must be set within message_description")));
}

TEST_F(FedSqlServerTest, StreamInitializeInvalidMessageDescriptorSetFails) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  init_config.mutable_private_logger_uploads_config()
      ->mutable_message_description()
      ->set_message_descriptor_set("invalid descriptor set");
  init_config.mutable_private_logger_uploads_config()
      ->mutable_message_description()
      ->set_message_name("some.message.Name");
  InitializeRequest request = CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse logged_message_descriptor_set.")));
}

TEST_F(FedSqlServerTest, StreamInitializeMessageNameNotFoundFails) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  MessageHelper message_helper;
  google::protobuf::FileDescriptorSet descriptor_set;
  message_helper.file_descriptor()->CopyTo(descriptor_set.add_file());
  init_config.mutable_private_logger_uploads_config()
      ->mutable_message_description()
      ->set_message_descriptor_set(descriptor_set.SerializeAsString());
  init_config.mutable_private_logger_uploads_config()
      ->mutable_message_description()
      ->set_message_name("some.nonexistent.Message");
  InitializeRequest request = CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Could not find message 'some.nonexistent.Message' "
                         "in the provided descriptor set.")));
}

TEST_F(FedSqlServerTest, StreamInitializeInvalidRequest) {
  FedSqlContainerInitializeConfiguration invalid_config;
  invalid_config.mutable_agg_configuration()
      ->add_intrinsic_configs()
      ->set_intrinsic_uri("BAD URI");
  InitializeRequest request =
      CreateInitializeRequest(std::move(invalid_config));
  InitializeResponse response;
  grpc::ClientContext context;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("is not a supported intrinsic_uri")));
}

TEST_F(FedSqlServerTest, StreamInitializeNoIntrinsicConfigs) {
  FedSqlContainerInitializeConfiguration invalid_config;
  InitializeRequest request =
      CreateInitializeRequest(std::move(invalid_config));
  InitializeResponse response;
  grpc::ClientContext context;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Configuration must have exactly one IntrinsicConfig")));
}

TEST_F(FedSqlServerTest, StreamInitializeFedSqlDpGroupByInvalidParameters) {
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
  InitializeRequest request = CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("must both have type DT_DOUBLE")));
}

TEST_F(FedSqlServerTest, StreamInitializeMultipleTopLevelIntrinsics) {
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
  InitializeRequest request = CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Configuration must have exactly one IntrinsicConfig")));
}

TEST_F(FedSqlServerTest, StreamInitializeMoreThanOnce) {
  InitializeRequest request = CreateInitializeRequest();
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), request), IsOk());

  grpc::ClientContext second_context;
  EXPECT_THAT(WriteInitializeRequest(
                  stub_->StreamInitialize(&second_context, &response),
                  std::move(request)),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("SetIntrinsics can only be called once")));
}

TEST_F(FedSqlServerTest, StreamInitializeDpConfigSuccess) {
  // Epsilon and delta defined in `DefaultFedSqlDpContainerConfig` are less than
  // the config_constraints below and so the request should succeed.
  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    epsilon: 1.2
    delta: 0.02
    intrinsic_uri: "fedsql_dp_group_by"
    access_budget { times: 5 }
  )pb");
  InitializeRequest request = CreateInitializeRequest(
      DefaultFedSqlDpContainerConfig(), std::move(config_constraints));
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  BudgetState budget_state =
      PARSE_TEXT_PROTO(R"pb(buckets { key: "foo" budget: 1 })pb");
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(),
                                        budget_state.SerializeAsString()));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());
}

TEST_F(FedSqlServerTest, StreamInitializeNoDpConfig) {
  InitializeRequest request = CreateInitializeRequest();
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());

  // Check that the write_configuration files exist for private state.
  ASSERT_TRUE(std::filesystem::exists("/tmp/write_configuration_1"));
  // Inference files shouldn't exist because no write_configuration is provided.
  ASSERT_FALSE(std::filesystem::exists("/tmp/write_configuration_2"));
}

TEST_F(FedSqlServerTest, StreamInitializeMultipleUrisSuccess) {
  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    epsilon: 1.1
    delta: 0.01
    intrinsic_uris: "some_other_uri"
    intrinsic_uris: "fedsql_dp_group_by"
    access_budget { times: 5 }
  )pb");
  InitializeRequest request = CreateInitializeRequest(
      DefaultFedSqlDpContainerConfig(), std::move(config_constraints));
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  BudgetState budget_state =
      PARSE_TEXT_PROTO(R"pb(buckets { key: "foo" budget: 1 })pb");
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(),
                                        budget_state.SerializeAsString()));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());
}

TEST_F(FedSqlServerTest, StreamInitializeInvalidPrivateState) {
  InitializeRequest request = CreateInitializeRequest();
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), "invalid private state"));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(FedSqlServerTest, StreamInitializeInvalidUri) {
  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    epsilon: 1.1
    delta: 0.01
    intrinsic_uris: "my_intrinsic_uri")pb");
  InitializeRequest request = CreateInitializeRequest(
      DefaultFedSqlDpContainerConfig(), std::move(config_constraints));
  InitializeResponse response;
  grpc::ClientContext context;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Invalid intrinsic URI for DP configuration.")));
}

TEST_F(FedSqlServerTest, StreamInitializeInvalidEpsilon) {
  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    epsilon: 0.8
    delta: 0.01
    intrinsic_uris: "fedsql_dp_group_by")pb");
  InitializeRequest request = CreateInitializeRequest(
      DefaultFedSqlDpContainerConfig(), std::move(config_constraints));
  InitializeResponse response;
  grpc::ClientContext context;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Epsilon value must be less than or equal to the "
                         "upper bound defined in the policy ")));
}

TEST_F(FedSqlServerTest, StreamInitializeInvalidDelta) {
  FedSqlContainerConfigConstraints config_constraints = PARSE_TEXT_PROTO(R"pb(
    epsilon: 1.1
    delta: 0.001
    intrinsic_uris: "fedsql_dp_group_by")pb");
  InitializeRequest request = CreateInitializeRequest(
      DefaultFedSqlDpContainerConfig(), std::move(config_constraints));
  InitializeResponse response;
  grpc::ClientContext context;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Delta value must be less than or equal to the "
                         "upper bound defined in the policy ")));
}

TEST_F(FedSqlServerTest, StreamInitializeInvalidConfigConstraints) {
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
  google::protobuf::Value value;
  InitializeRequest request = CreateInitializeRequest();
  InitializeResponse response;
  request.mutable_configuration()->PackFrom(value);
  grpc::ClientContext context;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Configuration cannot be unpacked.")));
}

TEST_F(FedSqlServerTest, StreamInitializeWithInferenceConfigs) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  InitializeResponse response;
  InitializeRequest initialize_request =
      CreateInitializeRequest(std::move(init_config));
  grpc::ClientContext context;

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

  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
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
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  *init_config.mutable_inference_init_config() = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
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
  InitializeRequest request = CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("model_init_config must be set.")));
}

TEST_F(FedSqlServerTest, StreamInitializeMissingModelConfig) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  *init_config.mutable_inference_init_config() = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
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
  InitializeRequest request = CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("model_config must be set.")));
}

TEST_F(FedSqlServerTest, StreamInitializeMissingInferenceLogic) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  *init_config.mutable_inference_init_config() = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
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
  InitializeRequest request = CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(request)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("inference_task.inference_logic must be set for all "
                         "inference tasks.")));
}

TEST_F(FedSqlServerTest,
       StreamInitializeWriteConfigurationRequestNotCommitted) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  InitializeRequest initialize_request =
      CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;
  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);

  StreamInitializeRequest write_configuration;
  ConfigurationMetadata* metadata =
      write_configuration.mutable_write_configuration()
          ->mutable_first_request_metadata();
  metadata->set_configuration_id("gemma_tokenizer_id");
  metadata->set_total_size_bytes(0);
  ASSERT_TRUE(writer->Write(write_configuration));

  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Data blob with configuration_id gemma_tokenizer_id is "
                    "not committed.")));
}

TEST_F(FedSqlServerTest, StreamInitializeInvalidGemmaTokenizerConfigurationId) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  InitializeRequest initialize_request =
      CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;
  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);

  StreamInitializeRequest write_configuration;
  ConfigurationMetadata* metadata =
      write_configuration.mutable_write_configuration()
          ->mutable_first_request_metadata();
  metadata->set_configuration_id("invalid_configuration_id");
  metadata->set_total_size_bytes(0);
  write_configuration.mutable_write_configuration()->set_commit(true);
  ASSERT_TRUE(writer->Write(write_configuration));

  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected gemma.cpp tokenizer configuration id "
              "gemma_tokenizer_id is missing in WriteConfigurationRequest.")));
}

TEST_F(FedSqlServerTest,
       StreamInitializeInvalidGemmaModelWeightConfigurationId) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  InitializeRequest initialize_request =
      CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;
  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);

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

  ASSERT_TRUE(writer->Write(tokenizer_write_config));
  ASSERT_TRUE(writer->Write(model_weight_write_config));
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected gemma.cpp weight configuration id "
                         "gemma_model_weight_id is missing in "
                         "WriteConfigurationRequest.")));
}

TEST_F(FedSqlServerTest,
       StreamInitializeWithGemmaInferenceSessionMissingTokenizerId) {
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
    inference_init_config {
      inference_config {
        inference_task: {
          column_config {
            input_column_names: [ "transcript" ]
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
          weight_type: WEIGHT_TYPE_SBS
        }
      }
      gemma_init_config {
        model_weight_configuration_id: "gemma_model_weight_id"
      }
    }
  )pb");
  InitializeRequest initialize_request =
      CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;
  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);

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

  ASSERT_TRUE(writer->Write(tokenizer_write_config));
  ASSERT_TRUE(writer->Write(first_model_weight_write_config));
  ASSERT_TRUE(writer->Write(second_model_weight_write_config));

  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Expected gemma.cpp tokenizer configuration id  is missing "
                    "in WriteConfigurationRequest")));

  // Remove inference files after assertions.
  std::filesystem::remove("/tmp/write_configuration_1");
  std::filesystem::remove("/tmp/write_configuration_2");
  std::filesystem::remove("/tmp/write_configuration_3");
}

TEST_F(FedSqlServerTest,
       StreamInitializeInvalidLlamaCppModelWeightConfigurationId) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  *init_config.mutable_inference_init_config() = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
      gemma_config {
        model_weight_file: "/path/to/model_weight"
        weight_type: WEIGHT_TYPE_GGUF
      }
    }
    llama_cpp_init_config {
      model_weight_configuration_id: "gemma_model_weight_id"
    }
  )pb");
  InitializeRequest initialize_request =
      CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;
  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);

  StreamInitializeRequest model_weight_write_config;
  ConfigurationMetadata* model_weight_metadata =
      model_weight_write_config.mutable_write_configuration()
          ->mutable_first_request_metadata();
  model_weight_metadata->set_configuration_id(
      "invalid_llama_cpp_model_weight_id");
  model_weight_metadata->set_total_size_bytes(0);
  model_weight_write_config.mutable_write_configuration()->set_commit(true);

  ASSERT_TRUE(writer->Write(model_weight_write_config));
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected llama.cpp weight configuration id "
                         "gemma_model_weight_id is missing in "
                         "WriteConfigurationRequest.")));
}

TEST_F(FedSqlServerTest, StreamInitializeDuplicatedConfigurationId) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  InitializeRequest initialize_request =
      CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;
  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);

  StreamInitializeRequest tokenizer_write_config;
  ConfigurationMetadata* tokenizer_metadata =
      tokenizer_write_config.mutable_write_configuration()
          ->mutable_first_request_metadata();
  tokenizer_metadata->set_configuration_id("gemma_tokenizer_id");
  tokenizer_metadata->set_total_size_bytes(0);
  tokenizer_write_config.mutable_write_configuration()->set_commit(true);

  ASSERT_TRUE(writer->Write(tokenizer_write_config));
  ASSERT_TRUE(writer->Write(tokenizer_write_config));
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Duplicated configuration_id found in "
                         "WriteConfigurationRequest")));
}

TEST_F(FedSqlServerTest, StreamInitializeInconsistentTotalSizeBytes) {
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  *init_config.mutable_inference_init_config() = DefaultInferenceConfig();
  InitializeRequest initialize_request =
      CreateInitializeRequest(std::move(init_config));
  InitializeResponse response;
  grpc::ClientContext context;
  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);

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

  ASSERT_TRUE(writer->Write(write_configuration));
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("The total size of the data blob does not match "
                         "expected size.")));
}

TEST_F(FedSqlServerTest, ConfigureSessionBeforeInitialize) {
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

TEST_F(FedSqlServerTest, ConfigureSessionSuccess) {
  InitializeTransform();

  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->set_chunk_size(1000);
  configure_request.mutable_configure()->mutable_configuration()->PackFrom(
      DefaultSqlQuery());
  auto stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_TRUE(stream->Read(&configure_response));
  ASSERT_TRUE(configure_response.has_configure());
}

TEST_F(FedSqlServerTest, ConfigureWriteReportEncryptedInput) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  // Write the encrypted request.
  std::string message_0 = BuildFedSqlGroupByCheckpoint({9}, {1});
  BlobHeader header;
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(1, 0)));
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  SessionRequest request_0 = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, message_0, header.SerializeAsString());
  SessionResponse response_0;
  ASSERT_TRUE(stream->Write(request_0));
  ASSERT_TRUE(stream->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), Code::OK);
  ASSERT_GT(response_0.write().committed_size_bytes(), 0);

  // Commit the request.
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  SessionRequest commit_request;
  SessionResponse commit_response;
  commit_request.mutable_commit()->mutable_configuration()->PackFrom(
      commit_config);
  ASSERT_TRUE(stream->Write(commit_request));
  EXPECT_TRUE(stream->Read(&commit_response));

  // Finalize the request.
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

  std::string decrypted_result =
      Decrypt(finalize_response.read().first_response_metadata(),
              finalize_response.read().data());
  ASSERT_TRUE(!decrypted_result.empty());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(decrypted_result);
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 2);
}

TEST_F(FedSqlServerTest, ConfigureInvalidRequest) {
  InitializeTransform();

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

TEST_F(FedSqlServerTest, ConfigureRequestWrongMessageType) {
  InitializeTransform();

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

TEST_F(FedSqlServerTest, ConfigureInvalidTableSchema) {
  InitializeTransform();

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

TEST_F(FedSqlServerTest, SessionRejectsMoreThanMaximumNumSessions) {
  InitializeTransform();

  std::vector<std::unique_ptr<
      grpc::ClientReaderWriter<SessionRequest, SessionResponse>>>
      streams;
  std::vector<std::unique_ptr<grpc::ClientContext>> contexts;
  for (int i = 0; i < kMaxNumSessions; i++) {
    std::unique_ptr<grpc::ClientContext> session_context =
        std::make_unique<grpc::ClientContext>();
    auto stream = ConfigureDefaultSession(session_context.get());

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

TEST_F(FedSqlServerTest, SessionFailsIfSqlResultCannotBeAggregated) {
  InitializeTransform();

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
  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->set_chunk_size(1000);
  configure_request.mutable_configure()->mutable_configuration()->PackFrom(
      query);
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_TRUE(stream->Read(&configure_response));

  BlobHeader header;
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(1, 0)));
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  SessionRequest write_request = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE,
      BuildFedSqlGroupByCheckpoint({7, 9}, {10, 12}),
      header.SerializeAsString());
  SessionResponse write_response;
  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));

  SessionRequest commit_request;
  SessionResponse commit_response;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  commit_request.mutable_commit()->mutable_configuration()->PackFrom(
      commit_config);
  ASSERT_TRUE(stream->Write(commit_request));
  ASSERT_FALSE(stream->Read(&commit_response));
  EXPECT_THAT(FromGrpcStatus(stream->Finish()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Failed to accumulate SQL query results")));
}

TEST_F(FedSqlServerTest, RemoveExpiredKeysFromBudget) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  // Write an encrypted request.
  BlobHeader header;
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(1, 0)));
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  auto request1 = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0}),
      header.SerializeAsString());
  SessionResponse response1;
  ASSERT_TRUE(stream->Write(request1));
  ASSERT_TRUE(stream->Read(&response1));
  ASSERT_TRUE(response1.has_write());

  // Commit the range.
  SessionRequest request2;
  SessionResponse response2;
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  request2.mutable_commit()->mutable_configuration()->PackFrom(commit_config);
  ASSERT_TRUE(stream->Write(request2));
  ASSERT_TRUE(stream->Read(&response2));
  ASSERT_TRUE(response2.has_commit());

  // Finalize the session which should remove `expired_key` from the budget.
  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse read_response;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&read_response));
  ASSERT_TRUE(read_response.has_read());
  ASSERT_TRUE(stream->Read(&finalize_response));
  ASSERT_TRUE(finalize_response.has_finalize());

  absl::StatusOr<ReleaseToken> release_token =
      ReleaseToken::Decode(finalize_response.finalize().release_token());
  ASSERT_THAT(release_token, IsOk());
  BudgetState new_state;
  EXPECT_TRUE(new_state.ParseFromString(release_token->dst_state.value()));
  EXPECT_THAT(new_state, EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                buckets {
                  key: "key_id"
                  budget: 4
                  consumed_range_start: 1
                  consumed_range_end: 3
                }
                buckets { key: "foo" budget: 3 }
              )pb"));
}

TEST_F(FedSqlServerTest, SessionWriteRequestNoKmsAssociatedData) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  SessionRequest request = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0}),
      "");
  SessionResponse response;
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_EQ(response.write().status().code(), Code::INVALID_ARGUMENT)
      << response.write().status().message();
}

TEST_F(FedSqlServerTest, SessionWriteRequestInvalidKmsAssociatedData) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  SessionRequest request = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0}),
      "invalid_associated_data");
  SessionResponse response;
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_EQ(response.write().status().code(), Code::INVALID_ARGUMENT)
      << response.write().status().message();
}

TEST_F(FedSqlServerTest, SessionWriteRequestInvalidPolicyHash) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  BlobHeader header;
  header.set_blob_id("blob_id");
  header.set_key_id(key_id_);
  header.set_access_policy_sha256("invalid_policy_hash");
  SessionRequest request = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0}),
      header.SerializeAsString());
  SessionResponse response;
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_EQ(response.write().status().code(), Code::INVALID_ARGUMENT)
      << response.write().status().message();
}

TEST_F(FedSqlServerTest, SessionWriteWithMetadataMessagesSuccess) {
  // Initialize the session.
  FedSqlContainerInitializeConfiguration init_config =
      DefaultFedSqlContainerConfig();
  MessageHelper message_helper;
  google::protobuf::FileDescriptorSet descriptor_set;
  message_helper.file_descriptor()->CopyTo(descriptor_set.add_file());
  fcp::confidentialcompute::PrivateLoggerUploadsConfig*
      private_logger_uploads_config =
          init_config.mutable_private_logger_uploads_config();
  private_logger_uploads_config->mutable_message_description()
      ->set_message_descriptor_set(descriptor_set.SerializeAsString());
  private_logger_uploads_config->mutable_message_description()
      ->set_message_name(message_helper.message_name());
  private_logger_uploads_config->set_on_device_query_name("test_query");
  InitializeRequest initialize_request =
      CreateInitializeRequest(std::move(init_config));
  InitializeResponse initialize_response;
  grpc::ClientContext context;
  auto writer = stub_->StreamInitialize(&context, &initialize_response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), ""));
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(initialize_request)),
      IsOk());

  // Configure the session.
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->set_chunk_size(1000);
  TableSchema schema = CreateTableSchema(
      "input",
      "CREATE TABLE input (key INTEGER, val INTEGER, "
      "confidential_compute_event_time STRING)",
      {CreateColumnSchema("key", google::internal::federated::plan::INT64),
       CreateColumnSchema("val", google::internal::federated::plan::INT64),
       CreateColumnSchema("confidential_compute_event_time",
                          google::internal::federated::plan::STRING)});
  SqlQuery query = CreateSqlQuery(
      schema, "SELECT key, val FROM input",
      {CreateColumnSchema("key", google::internal::federated::plan::INT64),
       CreateColumnSchema("val", google::internal::federated::plan::INT64)});
  configure_request.mutable_configure()->mutable_configuration()->PackFrom(
      query);
  grpc::ClientContext session_context_;
  auto stream = stub_->Session(&session_context_);
  CHECK(stream->Write(configure_request));
  CHECK(stream->Read(&configure_response));

  // Write the encrypted request.
  std::vector<std::string> serialized_messages;
  serialized_messages.push_back(
      message_helper.CreateMessage(8, 1)->SerializeAsString());
  std::vector<std::string> event_times = {"2023-01-01T00:00:00Z"};
  absl::StatusOr<std::string> message = BuildMessageCheckpoint(
      std::move(serialized_messages), std::move(event_times), "test_query");
  ASSERT_THAT(message, IsOk());

  BlobHeader header;
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(1, 0)));
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  SessionRequest request = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, *message, header.SerializeAsString());
  SessionResponse response;
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_EQ(response.write().status().code(), Code::OK)
      << response.write().status().message();
  ASSERT_GT(response.write().committed_size_bytes(), 0);
}

TEST_F(FedSqlServerTest, SessionExecutesSqlQueryAndAggregation) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  BlobHeader header;
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(1, 0)));
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  SessionRequest write_request_1 = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE,
      BuildFedSqlGroupByCheckpoint({1, 1, 2}, {1, 2, 5}),
      header.SerializeAsString());
  SessionResponse write_response_1;

  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_TRUE(stream->Read(&write_response_1));

  header.set_blob_id(StoreBigEndian(absl::MakeUint128(2, 0)));
  SessionRequest write_request_2 = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({1, 3}, {4, 0}),
      header.SerializeAsString());
  SessionResponse write_response_2;
  ASSERT_TRUE(stream->Write(write_request_2));
  ASSERT_TRUE(stream->Read(&write_response_2));

  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  SessionRequest commit_request;
  SessionResponse commit_response;
  commit_request.mutable_commit()->mutable_configuration()->PackFrom(
      commit_config);
  ASSERT_TRUE(stream->Write(commit_request));
  ASSERT_TRUE(stream->Read(&commit_response));

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
  std::string decrypted_data =
      Decrypt(finalize_response.read().first_response_metadata(),
              finalize_response.read().data());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(decrypted_data);
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The SQL query doubles each `val`, and the aggregation sums the val
  // column, grouping by key.
  ASSERT_EQ(col_values->num_elements(), 3);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  EXPECT_THAT(col_values->AsSpan<int64_t>(), UnorderedElementsAre(14, 10, 0));
}

TEST_F(FedSqlServerTest, SessionAccumulatesAndSerializes) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  // Write first encrypted request.
  BlobHeader header;
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(1, 0)));
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  SessionRequest write_request = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({9}, {1}),
      header.SerializeAsString());
  SessionResponse write_response;
  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));

  // Write second encrypted request.
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(2, 0)));
  write_request = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({9}, {1}),
      header.SerializeAsString());
  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));

  // Commit the range.
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  SessionRequest commit_request;
  SessionResponse commit_response;
  commit_request.mutable_commit()->mutable_configuration()->PackFrom(
      commit_config);
  ASSERT_TRUE(stream->Write(commit_request));
  ASSERT_TRUE(stream->Read(&commit_response));

  // Finalize the session.
  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
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

  // Decrypt the result.
  std::string decrypted_data =
      Decrypt(finalize_response.read().first_response_metadata(),
              finalize_response.read().data());
  auto range_tracker = UnbundleRangeTracker(decrypted_data);
  ASSERT_THAT(*range_tracker,
              UnorderedElementsAre(
                  Pair("key_id", ElementsAre(Interval<uint64_t>(1, 3)))));

  // Validate the deserialized aggregator.
  Configuration intrinsic_config =
      DefaultFedSqlContainerConfig().agg_configuration();
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> deserialized_agg =
      CheckpointAggregator::Deserialize(intrinsic_config, decrypted_data);
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

TEST_F(FedSqlServerTest, SessionMergesAndReports) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  FederatedComputeCheckpointParserFactory parser_factory;
  auto input_parser =
      parser_factory.Create(absl::Cord(BuildFedSqlGroupByCheckpoint({4}, {3})))
          .value();
  std::unique_ptr<CheckpointAggregator> input_aggregator =
      CheckpointAggregator::Create(
          DefaultFedSqlContainerConfig().agg_configuration())
          .value();
  ASSERT_THAT(input_aggregator->Accumulate(*input_parser), IsOk());
  std::string data = std::move(*input_aggregator).Serialize().value();
  RangeTracker range_tracker;
  range_tracker.AddRange("key_id", 1, 3);
  std::string blob = BundleRangeTracker(data, range_tracker);

  BlobHeader header;
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(10, 0)));
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  SessionRequest write_request = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_MERGE, blob, header.SerializeAsString());
  SessionResponse write_response;

  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));

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

  std::string decrypted_data =
      Decrypt(finalize_response.read().first_response_metadata(),
              finalize_response.read().data());
  absl::Cord wire_format_result(decrypted_data);
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The input aggregator should be merged with the session aggregator
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 3);
}

TEST_F(FedSqlServerTest, SerializeZeroInputsProducesEmptyOutput) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_SERIALIZE
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

  std::string data = Decrypt(finalize_response.read().first_response_metadata(),
                             finalize_response.read().data());
  auto ignored_range_tracker = UnbundleRangeTracker(data);
  absl::StatusOr<std::unique_ptr<CheckpointAggregator>>
      deserialized_agg_status = CheckpointAggregator::Deserialize(
          DefaultFedSqlContainerConfig().agg_configuration(), data);
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
      CheckpointAggregator::Create(
          DefaultFedSqlContainerConfig().agg_configuration())
          .value();
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

TEST_F(FedSqlServerTest, ReportZeroInputsReturnsInvalidArgument) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  FedSqlContainerFinalizeConfiguration finalize_config = PARSE_TEXT_PROTO(R"pb(
    type: FINALIZATION_TYPE_REPORT
  )pb");
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_FALSE(stream->Read(&finalize_response));
  EXPECT_THAT(FromGrpcStatus(stream->Finish()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(FedSqlServerTest, SessionIgnoresUnparseableInputs) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  // Write a valid encrypted request.
  BlobHeader header;
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(1, 0)));
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  SessionRequest write_request_1 = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({8}, {7}),
      header.SerializeAsString());
  SessionResponse write_response_1;
  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_TRUE(stream->Read(&write_response_1));

  // Write an invalid encrypted request.
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(2, 0)));
  SessionRequest invalid_write = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, "invalid checkpoint",
      header.SerializeAsString());
  SessionResponse invalid_write_response;
  ASSERT_TRUE(stream->Write(invalid_write));
  ASSERT_TRUE(stream->Read(&invalid_write_response));
  ASSERT_TRUE(invalid_write_response.has_write());
  ASSERT_EQ(invalid_write_response.write().committed_size_bytes(), 0);
  ASSERT_EQ(invalid_write_response.write().status().code(),
            Code::INVALID_ARGUMENT)
      << invalid_write_response.write().status().message();

  // Commit the range.
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  SessionRequest commit_request;
  SessionResponse commit_response;
  commit_request.mutable_commit()->mutable_configuration()->PackFrom(
      commit_config);
  ASSERT_TRUE(stream->Write(commit_request));
  ASSERT_TRUE(stream->Read(&commit_response));

  // Finalize the session.
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
  std::string decrypted_data =
      Decrypt(finalize_response.read().first_response_metadata(),
              finalize_response.read().data());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(decrypted_data);
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The invalid input should be ignored
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 14);
}

TEST_F(FedSqlServerTest, SessionIgnoresInputThatCannotBeQueried) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  BlobHeader header;
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(1, 0)));
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  SessionRequest write_request_1 = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE,
      BuildFedSqlGroupByCheckpoint({9}, {7}, "bad_key_col_name"),
      header.SerializeAsString());
  SessionResponse write_response_1;

  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_TRUE(stream->Read(&write_response_1));
  ASSERT_EQ(write_response_1.write().status().code(), Code::NOT_FOUND)
      << write_response_1.write().status().message();
}

TEST_F(FedSqlServerTest, SessionIgnoresUndecryptableInputs) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = ConfigureDefaultSession(&context);

  // Write a valid encrypted request.
  BlobHeader header;
  header.set_blob_id(StoreBigEndian(absl::MakeUint128(1, 0)));
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  SessionRequest write_request_1 = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({42}, {2}),
      header.SerializeAsString());
  SessionResponse write_response_1;
  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_TRUE(stream->Read(&write_response_1));

  // Write an invalid encrypted request.
  SessionRequest invalid_write = CreateDefaultEncryptedWriteRequest(
      AGGREGATION_TYPE_ACCUMULATE, BuildFedSqlGroupByCheckpoint({42}, {2}),
      "invalid associated data");
  SessionResponse invalid_write_response;
  ASSERT_TRUE(stream->Write(invalid_write));
  ASSERT_TRUE(stream->Read(&invalid_write_response));
  ASSERT_TRUE(invalid_write_response.has_write());
  ASSERT_EQ(invalid_write_response.write().committed_size_bytes(), 0);
  ASSERT_EQ(invalid_write_response.write().status().code(),
            Code::INVALID_ARGUMENT)
      << invalid_write_response.write().status().message();

  // Commit the range.
  FedSqlContainerCommitConfiguration commit_config = PARSE_TEXT_PROTO(R"pb(
    range { start: 1 end: 3 }
  )pb");
  SessionRequest commit_request;
  SessionResponse commit_response;
  commit_request.mutable_commit()->mutable_configuration()->PackFrom(
      commit_config);
  ASSERT_TRUE(stream->Write(commit_request));
  ASSERT_TRUE(stream->Read(&commit_response));

  // Finalize the session.
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
  std::string decrypted_data =
      Decrypt(finalize_response.read().first_response_metadata(),
              finalize_response.read().data());

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord wire_format_result(decrypted_data);
  auto parser = parser_factory.Create(wire_format_result);
  auto col_values = (*parser)->GetTensor("val_out");
  // The undecryptable write is ignored, and only the valid write is aggregated.
  ASSERT_EQ(col_values->num_elements(), 1);
  ASSERT_EQ(col_values->dtype(), DataType::DT_INT64);
  ASSERT_EQ(col_values->AsSpan<int64_t>().at(0), 4);
}

}  // namespace

}  // namespace confidential_federated_compute::fed_sql
