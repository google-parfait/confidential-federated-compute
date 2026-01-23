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
#include "containers/confidential_transform_server_base.h"

#include <memory>
#include <string>
#include <tuple>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "cc/crypto/client_encryptor.h"
#include "cc/crypto/encryption_key.h"
#include "cc/crypto/signing_key.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute {

namespace {

using ::absl_testing::IsOk;
using ::fcp::base::FromGrpcStatus;
using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigurationMetadata;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::FinalizeResponse;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::oak::crypto::ClientEncryptor;
using ::oak::crypto::EncryptionKeyProvider;
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::Test;

using SessionStream =
    ::grpc::ClientReaderWriter<SessionRequest, SessionResponse>;

inline constexpr int kMaxNumSessions = 8;
constexpr absl::string_view kKeyId = "key_id";

class MockSession final : public confidential_federated_compute::Session {
 public:
  MOCK_METHOD(absl::StatusOr<ConfigureResponse>, Configure,
              (ConfigureRequest request, Context& context), (override));
  MOCK_METHOD(absl::StatusOr<WriteFinishedResponse>, Write,
              (WriteRequest request, std::string unencrypted_data,
               Context& context),
              (override));
  MOCK_METHOD(absl::StatusOr<CommitResponse>, Commit,
              (CommitRequest request, Context& context), (override));
  MOCK_METHOD(absl::StatusOr<FinalizeResponse>, Finalize,
              (FinalizeRequest request, BlobMetadata input_metadata,
               Context& context),
              (override));
};

SessionRequest CreateDefaultWriteRequest(std::string data) {
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  metadata.set_total_size_bytes(data.size());
  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->set_commit(true);
  write_request->set_data(data);
  return request;
}

ReadResponse GetFinalizeReadResponse(const std::string& result) {
  ReadResponse read_response;
  read_response.set_finish_read(true);
  *(read_response.mutable_data()) = result;
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  metadata.set_total_size_bytes(result.size());
  *(read_response.mutable_first_response_metadata()) = metadata;
  return read_response;
}

ReadResponse GetDefaultFinalizeReadResponse() {
  return GetFinalizeReadResponse("test result");
}

class FakeConfidentialTransform final : public ConfidentialTransformBase {
 public:
  FakeConfidentialTransform(
      std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle,
      std::unique_ptr<oak::crypto::EncryptionKeyHandle> encryption_key_handle)
      : ConfidentialTransformBase(std::move(signing_key_handle),
                                  std::move(encryption_key_handle)) {}

  void AddSession(
      std::unique_ptr<confidential_federated_compute::Session> session) {
    session_ = std::move(session);
  };

  absl::flat_hash_set<std::string> GetActiveKeyIds() const {
    return ConfidentialTransformBase::GetActiveKeyIds();
  }

  bool ActiveKeyIdsIncludeAllKeysets() const {
    return ConfidentialTransformBase::ActiveKeyIdsIncludeAllKeysets();
  }

 protected:
  absl::Status StreamInitializeTransform(
      const ::google::protobuf::Any& configuration,
      const ::google::protobuf::Any& config_constraints) override {
    google::rpc::Status config_status;
    if (!configuration.UnpackTo(&config_status)) {
      return absl::InvalidArgumentError("Config cannot be unpacked.");
    }
    if (config_status.code() != grpc::StatusCode::OK) {
      return absl::InvalidArgumentError("Invalid config.");
    }
    return absl::OkStatus();
  }

  absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override {
    if (session_ == nullptr) {
      auto session = std::make_unique<MockSession>();
      EXPECT_CALL(*session, Configure).WillOnce(Return(ConfigureResponse{}));
      return std::move(session);
    }
    return std::move(session_);
  }

  absl::StatusOr<std::string> GetKeyId(
      const fcp::confidentialcompute::BlobMetadata& metadata) override {
    return std::string(kKeyId);
  }

 private:
  std::unique_ptr<confidential_federated_compute::Session> session_;
};

class ConfidentialTransformServerBaseTest : public Test {
 public:
  ConfidentialTransformServerBaseTest() {
    int port;
    const std::string server_address = "[::1]:";
    auto encryption_key_handle = std::make_unique<EncryptionKeyProvider>(
        EncryptionKeyProvider::Create().value());
    oak_client_encryptor_ =
        ClientEncryptor::Create(encryption_key_handle->GetSerializedPublicKey())
            .value();
    service_ = std::make_unique<FakeConfidentialTransform>(
        std::make_unique<
            testing::NiceMock<crypto_test_utils::MockSigningKeyHandle>>(),
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

  ~ConfidentialTransformServerBaseTest() override { server_->Shutdown(); }

 protected:
  StreamInitializeRequest CreateStreamInitializeRequest() {
    google::rpc::Status config_status;
    config_status.set_code(grpc::StatusCode::OK);
    auto public_private_key_pair =
        crypto_test_utils::GenerateKeyPair(std::string(kKeyId));
    public_key_ = public_private_key_pair.first;
    AuthorizeConfidentialTransformResponse::ProtectedResponse
        protected_response;
    *protected_response.add_result_encryption_keys() =
        public_private_key_pair.first;
    *protected_response.add_decryption_keys() = public_private_key_pair.second;
    AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
    associated_data.add_authorized_logical_pipeline_policies_hashes(
        "policy_hash");
    associated_data.add_omitted_decryption_key_ids("omitted_key_id");
    associated_data.set_omitted_decryption_key_ids_include_all_keysets(true);
    auto encrypted_request =
        oak_client_encryptor_
            ->Encrypt(protected_response.SerializeAsString(),
                      associated_data.SerializeAsString())
            .value();

    StreamInitializeRequest request;
    request.mutable_initialize_request()->mutable_configuration()->PackFrom(
        config_status);
    request.mutable_initialize_request()->set_max_num_sessions(kMaxNumSessions);
    *request.mutable_initialize_request()->mutable_protected_response() =
        encrypted_request;
    return request;
  }

  void InitializeTransform() {
    auto request = CreateStreamInitializeRequest();
    grpc::ClientContext configure_context;
    InitializeResponse response;
    std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
        stub_->StreamInitialize(&configure_context, &response);
    CHECK(writer->Write(request));
    CHECK(writer->WritesDone());
    CHECK(writer->Finish().ok());
    auto key_ids = service_->GetActiveKeyIds();
    CHECK_EQ(key_ids.size(), 2);
    CHECK(key_ids.contains(kKeyId));
    CHECK(key_ids.contains("omitted_key_id"));
  }

  std::unique_ptr<SessionStream> StartSession(grpc::ClientContext* context,
                                              uint32_t chunk_size = 1000) {
    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure()->set_chunk_size(chunk_size);

    auto stream = stub_->Session(context);
    CHECK(stream->Write(session_request));
    CHECK(stream->Read(&session_response));
    return stream;
  }

  std::pair<BlobMetadata, std::string> Encrypt(
      std::string blob_id, std::string message,
      std::string policy_hash = "policy_hash") {
    BlobHeader header;
    header.set_blob_id(blob_id);
    header.set_key_id(std::string(kKeyId));
    header.set_access_policy_sha256(policy_hash);
    std::string associated_data = header.SerializeAsString();

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

  std::unique_ptr<FakeConfidentialTransform> service_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
  std::unique_ptr<ClientEncryptor> oak_client_encryptor_;
  std::string public_key_;
};

TEST_F(ConfidentialTransformServerBaseTest, ValidStreamInitialize) {
  grpc::ClientContext context;
  InitializeResponse response;

  // A write_configuration request where the entire blob can be passed in using
  // one message. Therefore both first_request_metadata and commit are set.
  StreamInitializeRequest write_configuration;
  ConfigurationMetadata* metadata =
      write_configuration.mutable_write_configuration()
          ->mutable_first_request_metadata();
  metadata->set_configuration_id("configuration_id");
  metadata->set_total_size_bytes(100);
  write_configuration.mutable_write_configuration()->set_commit(true);

  // An initialize_request.
  StreamInitializeRequest initialize_request = CreateStreamInitializeRequest();

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(write_configuration));
  ASSERT_TRUE(writer->Write(initialize_request));
  // WritesDone is called to indicate that no more messages will be sent.
  ASSERT_TRUE(writer->WritesDone());
  // Finish is called to get the server's response and final status of the
  // stream.
  ASSERT_THAT(FromGrpcStatus(writer->Finish()), IsOk());
  ASSERT_TRUE(service_->ActiveKeyIdsIncludeAllKeysets());
}

TEST_F(ConfidentialTransformServerBaseTest,
       ValidStreamInitializeNoWriteConfiguration) {
  grpc::ClientContext context;
  InitializeResponse response;
  StreamInitializeRequest initialize_request = CreateStreamInitializeRequest();

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(initialize_request));
  ASSERT_TRUE(writer->WritesDone());
  ASSERT_THAT(FromGrpcStatus(writer->Finish()), IsOk());
}

TEST_F(ConfidentialTransformServerBaseTest,
       StreamInitializeInitializeRequestBeforeWriteConfiguration) {
  grpc::ClientContext context;
  InitializeResponse response;

  StreamInitializeRequest initialize_request = CreateStreamInitializeRequest();
  StreamInitializeRequest write_configuration;
  ConfigurationMetadata* metadata =
      write_configuration.mutable_write_configuration()
          ->mutable_first_request_metadata();
  metadata->set_configuration_id("configuration_id");
  metadata->set_total_size_bytes(100);
  write_configuration.mutable_write_configuration()->set_commit(true);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(initialize_request));
  ASSERT_TRUE(writer->Write(write_configuration));
  ASSERT_TRUE(writer->WritesDone());
  auto status = writer->Finish();
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(
      status.error_message(),
      HasSubstr(
          "Expect all StreamInitializeRequests.write_configurations to be "
          "sent before the StreamInitializeRequest.initialize_request."));
}

TEST_F(ConfidentialTransformServerBaseTest,
       StreamInitializeRequestWrongMessageType) {
  grpc::ClientContext context;
  google::protobuf::Value value;
  InitializeResponse response;
  StreamInitializeRequest request = CreateStreamInitializeRequest();
  request.mutable_initialize_request()->mutable_configuration()->PackFrom(
      value);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(request));
  ASSERT_TRUE(writer->WritesDone());
  auto status = writer->Finish();
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  ASSERT_THAT(status.error_message(), HasSubstr("Config cannot be unpacked."));
}

TEST_F(ConfidentialTransformServerBaseTest,
       StreamInitializeInvalidRequestKind) {
  grpc::ClientContext context;
  StreamInitializeRequest request;
  InitializeResponse response;

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(request));
  ASSERT_TRUE(writer->WritesDone());
  auto status = writer->Finish();
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(), HasSubstr("received request of type: 0"));
}

TEST_F(ConfidentialTransformServerBaseTest, StreamInitializeMoreThanOnce) {
  grpc::ClientContext context;
  InitializeResponse response;
  StreamInitializeRequest request = CreateStreamInitializeRequest();

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(request));
  ASSERT_TRUE(writer->WritesDone());
  ASSERT_THAT(FromGrpcStatus(writer->Finish()), IsOk());

  grpc::ClientContext second_context;
  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> second_writer =
      stub_->StreamInitialize(&second_context, &response);
  ASSERT_TRUE(second_writer->Write(request));
  ASSERT_TRUE(second_writer->WritesDone());
  auto status = second_writer->Finish();

  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("StreamInitialize can only be called once"));
}

TEST_F(ConfidentialTransformServerBaseTest,
       StreamInitializeOnlyWriteConfiguration) {
  grpc::ClientContext context;
  InitializeResponse response;

  StreamInitializeRequest write_configuration;
  ConfigurationMetadata* metadata =
      write_configuration.mutable_write_configuration()
          ->mutable_first_request_metadata();
  metadata->set_configuration_id("configuration_id");
  metadata->set_total_size_bytes(100);
  write_configuration.mutable_write_configuration()->set_commit(true);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(write_configuration));
  ASSERT_TRUE(writer->WritesDone());
  auto status = writer->Finish();

  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("Expect one of the StreamInitializeRequests to be "
                        "configured with a InitializeRequest, found zero."));
}

TEST_F(ConfidentialTransformServerBaseTest,
       StreamInitializeMoreThanOneInitializeRequest) {
  grpc::ClientContext context;
  InitializeResponse response;
  StreamInitializeRequest initialize_request = CreateStreamInitializeRequest();

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(initialize_request));
  ASSERT_TRUE(writer->Write(initialize_request));
  ASSERT_TRUE(writer->WritesDone());
  auto status = writer->Finish();

  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(
      status.error_message(),
      HasSubstr("Expect one of the StreamInitializeRequests to be "
                "configured with a InitializeRequest, found more than one."));
}

TEST_F(ConfidentialTransformServerBaseTest, SessionBeforeInitialize) {
  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->set_chunk_size(1000);
  configure_request.mutable_configure()->mutable_configuration();

  std::unique_ptr<SessionStream> stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_FALSE(stream->Read(&configure_response));
  auto status = stream->Finish();
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("StreamInitialize must be called before Session"));
}

TEST_F(ConfidentialTransformServerBaseTest,
       StreamInitializeAndConfigureSession) {
  InitializeTransform();
  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_configure()->set_chunk_size(1000);

  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce([](FinalizeRequest request, BlobMetadata unused,
                   Session::Context& context) {
        context.Emit(GetDefaultFinalizeReadResponse());
        return FinalizeResponse{};
      });
  service_->AddSession(std::move(mock_session));

  std::unique_ptr<SessionStream> stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_TRUE(stream->Read(&session_response));
  ASSERT_TRUE(session_response.has_configure());

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  SessionRequest finalize_request;
  SessionResponse read_response, finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&read_response));
  ASSERT_TRUE(stream->Read(&finalize_response));
  ASSERT_THAT(FromGrpcStatus(stream->Finish()), IsOk());
}

TEST_F(ConfidentialTransformServerBaseTest, ChunkSizeNotSpecified) {
  InitializeTransform();
  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->mutable_configuration();

  std::unique_ptr<SessionStream> stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_FALSE(stream->Read(&configure_response));
  auto status = stream->Finish();
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("chunk_size must be specified"));
}

TEST_F(ConfidentialTransformServerBaseTest,
       StreamInitializeSessionRejectsMoreThanMaximumNumSessions) {
  InitializeTransform();

  std::vector<std::unique_ptr<SessionStream>> streams;
  std::vector<std::unique_ptr<grpc::ClientContext>> contexts;
  for (int i = 0; i < kMaxNumSessions; i++) {
    std::unique_ptr<grpc::ClientContext> session_context =
        std::make_unique<grpc::ClientContext>();
    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure()->set_chunk_size(1000);

    std::unique_ptr<SessionStream> stream =
        stub_->Session(session_context.get());
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

  std::unique_ptr<SessionStream> stream = stub_->Session(&rejected_context);
  ASSERT_TRUE(stream->Write(rejected_request));
  ASSERT_FALSE(stream->Read(&rejected_response));
  ASSERT_EQ(stream->Finish().error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(ConfidentialTransformServerBaseTest,
       SessionWriteFinalizeUnencryptedBlob) {
  InitializeTransform();
  std::string data = "test data";
  SessionRequest write_request = CreateDefaultWriteRequest(data);
  SessionResponse write_response;

  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Write)
      .WillRepeatedly(
          Return(ToWriteFinishedResponse(absl::OkStatus(), data.size())));
  EXPECT_CALL(*mock_session, Commit)
      .WillRepeatedly(Return(ToCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce([](FinalizeRequest request, BlobMetadata unused,
                   Session::Context& context) {
        context.Emit(GetDefaultFinalizeReadResponse());
        return FinalizeResponse{};
      });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  // Accumulate the same unencrypted blob twice.
  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));
  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  SessionRequest finalize_request;
  SessionResponse read_response, finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&read_response));
  ASSERT_TRUE(stream->Read(&finalize_response));
  ASSERT_THAT(FromGrpcStatus(stream->Finish()), IsOk());

  ASSERT_THAT(read_response,
              EqualsProto(R"pb(read {
                                 first_response_metadata {
                                   total_size_bytes: 11
                                   compression_type: COMPRESSION_TYPE_NONE
                                   unencrypted {}
                                 }
                                 finish_read: true
                                 data: "test result"
                               })pb"));
  ASSERT_THAT(finalize_response, EqualsProto(R"pb(finalize {})pb"));
}

TEST_F(ConfidentialTransformServerBaseTest, SessionWriteFinalizeEncryptedBlob) {
  InitializeTransform();
  std::string message = "test data";
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Write)
      .WillOnce(
          Return(ToWriteFinishedResponse(absl::OkStatus(), message.size())));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce([](FinalizeRequest request, BlobMetadata unused,
                   Session::Context& context) {
        context.Emit(GetDefaultFinalizeReadResponse());
        return FinalizeResponse{};
      });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  auto [metadata, ciphertext] = Encrypt("blob_id", message);
  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->mutable_first_request_configuration()->PackFrom(config);
  write_request->set_commit(true);
  write_request->set_data(ciphertext);

  SessionResponse response;
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_EQ(response.write().status().code(), grpc::OK);
  ASSERT_EQ(response.write().committed_size_bytes(), message.size());

  SessionRequest finalize_request;
  SessionResponse read_response, finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&read_response));
  ASSERT_TRUE(stream->Read(&finalize_response));

  ASSERT_THAT(read_response,
              EqualsProto(R"pb(read {
                                 first_response_metadata {
                                   total_size_bytes: 11
                                   compression_type: COMPRESSION_TYPE_NONE
                                   unencrypted {}
                                 }
                                 finish_read: true
                                 data: "test result"
                               })pb"));
  ASSERT_THAT(finalize_response, EqualsProto(R"pb(finalize {})pb"));
}

TEST_F(ConfidentialTransformServerBaseTest,
       SessionWritesChunkedBlobAndFinalizes) {
  InitializeTransform();

  std::string chunk1 = "abcdef";
  std::string chunk2 = "ghijkl";
  std::string chunk3 = "mno";
  std::string data = chunk1 + chunk2 + chunk3;

  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Write(_, data, _))
      .WillOnce(Return(ToWriteFinishedResponse(absl::OkStatus(), data.size())));
  EXPECT_CALL(*mock_session, Commit)
      .WillRepeatedly(Return(ToCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce([](FinalizeRequest request, BlobMetadata unused,
                   Session::Context& context) {
        context.Emit(GetDefaultFinalizeReadResponse());
        return FinalizeResponse{};
      });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  metadata.set_total_size_bytes(data.size());
  SessionRequest write_request1;
  *write_request1.mutable_write()->mutable_first_request_metadata() = metadata;
  write_request1.mutable_write()->set_data(chunk1);
  ASSERT_TRUE(stream->Write(write_request1));

  SessionRequest write_request2;
  write_request2.mutable_write()->set_data(chunk2);
  ASSERT_TRUE(stream->Write(write_request2));

  SessionRequest write_request3;
  write_request3.mutable_write()->set_data(chunk3);
  write_request3.mutable_write()->set_commit(true);
  ASSERT_TRUE(stream->Write(write_request3));

  SessionResponse write_response;
  ASSERT_TRUE(stream->Read(&write_response));

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  SessionRequest finalize_request;
  SessionResponse read_response, finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&read_response));
  ASSERT_TRUE(stream->Read(&finalize_response));
  ASSERT_THAT(FromGrpcStatus(stream->Finish()), IsOk());

  ASSERT_THAT(read_response,
              EqualsProto(R"pb(read {
                                 first_response_metadata {
                                   total_size_bytes: 11
                                   compression_type: COMPRESSION_TYPE_NONE
                                   unencrypted {}
                                 }
                                 finish_read: true
                                 data: "test result"
                               })pb"));
  ASSERT_THAT(finalize_response, EqualsProto(R"pb(finalize {})pb"));
}

TEST_F(ConfidentialTransformServerBaseTest,
       SessionFinalizesWithChunkedReadResponse) {
  InitializeTransform();
  std::string data = "test data";
  SessionRequest write_request = CreateDefaultWriteRequest(data);
  SessionResponse write_response;

  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce([](FinalizeRequest request, BlobMetadata unused,
                   Session::Context& context) {
        context.Emit(GetFinalizeReadResponse("abcdefghijklmno"));
        return FinalizeResponse{};
      });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context, /*chunk_size=*/6);

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  SessionRequest finalize_request;
  SessionResponse read_response1, read_response2, read_response3,
      finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&read_response1));
  ASSERT_TRUE(stream->Read(&read_response2));
  ASSERT_TRUE(stream->Read(&read_response3));
  ASSERT_TRUE(stream->Read(&finalize_response));
  ASSERT_THAT(FromGrpcStatus(stream->Finish()), IsOk());

  ASSERT_THAT(read_response1,
              EqualsProto(R"pb(read {
                                 first_response_metadata {
                                   total_size_bytes: 15
                                   compression_type: COMPRESSION_TYPE_NONE
                                   unencrypted {}
                                 }
                                 data: "abcdef"
                               })pb"));
  ASSERT_THAT(read_response2, EqualsProto(R"pb(read { data: "ghijkl" })pb"));
  ASSERT_THAT(read_response3,
              EqualsProto(R"pb(read { data: "mno" finish_read: true })pb"));
  ASSERT_THAT(finalize_response, EqualsProto(R"pb(finalize {})pb"));
}

TEST_F(ConfidentialTransformServerBaseTest, SessionIgnoresInvalidInputs) {
  InitializeTransform();
  std::string data = "test data";
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Write)
      .WillOnce(Return(ToWriteFinishedResponse(absl::OkStatus(), data.size())))
      .WillOnce(Return(ToWriteFinishedResponse(
          absl::InvalidArgumentError("Invalid argument"), 0)));
  EXPECT_CALL(*mock_session, Commit)
      .WillRepeatedly(Return(ToCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce([](FinalizeRequest request, BlobMetadata unused,
                   Session::Context& context) {
        context.Emit(GetDefaultFinalizeReadResponse());
        return FinalizeResponse{};
      });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest write_request_1 = CreateDefaultWriteRequest(data);
  SessionResponse write_response_1;

  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_TRUE(stream->Read(&write_response_1));

  SessionRequest write_request_2 = CreateDefaultWriteRequest(data);
  SessionResponse write_response_2;

  ASSERT_TRUE(stream->Write(write_request_2));
  ASSERT_TRUE(stream->Read(&write_response_2));

  ASSERT_TRUE(write_response_2.has_write());
  ASSERT_EQ(write_response_2.write().committed_size_bytes(), 0);
  ASSERT_EQ(write_response_2.write().status().code(), grpc::INVALID_ARGUMENT);

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  SessionRequest finalize_request;
  SessionResponse read_response, finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&read_response));
  ASSERT_TRUE(stream->Read(&finalize_response));

  ASSERT_THAT(read_response,
              EqualsProto(R"pb(read {
                                 first_response_metadata {
                                   total_size_bytes: 11
                                   compression_type: COMPRESSION_TYPE_NONE
                                   unencrypted {}
                                 }
                                 finish_read: true
                                 data: "test result"
                               })pb"));
  ASSERT_THAT(finalize_response, EqualsProto(R"pb(finalize {})pb"));
}

TEST_F(ConfidentialTransformServerBaseTest, SessionFailsIfWriteFails) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Write)
      .WillOnce(Return(absl::InternalError("Internal Error")));
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest write_request_1 = CreateDefaultWriteRequest("test data");
  SessionResponse write_response_1;

  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_FALSE(stream->Read(&write_response_1));
  ASSERT_EQ(stream->Finish().error_code(), grpc::StatusCode::INTERNAL);
}

TEST_F(ConfidentialTransformServerBaseTest,
       SessionFailsIfIncompleteChunkedBlobWriteFollowedByWrite) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest write_request_1 = CreateDefaultWriteRequest("test data");
  SessionRequest write_request_2 = CreateDefaultWriteRequest("test data");
  SessionResponse write_response;

  write_request_1.mutable_write()->set_commit(false);
  ASSERT_TRUE(stream->Write(write_request_1));
  ASSERT_TRUE(stream->Write(write_request_2));
  ASSERT_FALSE(stream->Read(&write_response));
  ASSERT_EQ(stream->Finish().error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(ConfidentialTransformServerBaseTest,
       SessionFailsIfIncompleteChunkedBlobWriteFollowedByFinalize) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest write_request = CreateDefaultWriteRequest("test data");
  SessionRequest finalize_request;
  SessionResponse finalize_response;

  write_request.mutable_write()->set_commit(false);
  ASSERT_TRUE(stream->Write(write_request));

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_FALSE(stream->Read(&finalize_response));
  ASSERT_EQ(stream->Finish().error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(ConfidentialTransformServerBaseTest, SessionFailsIfCommitFails) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Commit)
      .WillOnce(Return(absl::InternalError("Internal Error")));
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest commit_request;
  commit_request.mutable_commit();
  SessionResponse commit_response;

  ASSERT_TRUE(stream->Write(commit_request));
  ASSERT_FALSE(stream->Read(&commit_response));
  ASSERT_EQ(stream->Finish().error_code(), grpc::StatusCode::INTERNAL);
}

TEST_F(ConfidentialTransformServerBaseTest, SessionFailsIfFinalizeFails) {
  InitializeTransform();
  std::string data = "test data";
  SessionRequest write_request = CreateDefaultWriteRequest(data);
  SessionResponse write_response;

  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Write)
      .WillRepeatedly(
          Return(ToWriteFinishedResponse(absl::OkStatus(), data.size())));

  EXPECT_CALL(*mock_session, Commit)
      .WillRepeatedly(Return(ToCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce(Return(absl::InternalError("Internal Error")));
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  // Accumulate the same unencrypted blob twice.
  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));
  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::INTERNAL);
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_FALSE(stream->Read(&finalize_response));
  ASSERT_EQ(stream->Finish().error_code(), grpc::StatusCode::INTERNAL);
}

TEST_F(ConfidentialTransformServerBaseTest,
       TransformIgnoresUndecryptableInputs) {
  InitializeTransform();
  std::string message = "test data";
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Write)
      .WillOnce(
          Return(ToWriteFinishedResponse(absl::OkStatus(), message.size())));
  EXPECT_CALL(*mock_session, Commit)
      .WillRepeatedly(Return(ToCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce([](FinalizeRequest request, BlobMetadata unused,
                   Session::Context& context) {
        context.Emit(GetDefaultFinalizeReadResponse());
        return FinalizeResponse{};
      });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  // Create one blob that will fail to decrypt and one blob that can be
  // successfully decrypted.
  auto [metadata, ciphertext] = Encrypt("blob_id", message);
  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->mutable_first_request_configuration()->PackFrom(config);
  write_request->set_commit(true);
  write_request->set_data(ciphertext);

  SessionResponse response;
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_EQ(response.write().status().code(), grpc::OK);
  ASSERT_EQ(response.write().committed_size_bytes(), message.size());

  metadata.mutable_hpke_plus_aead_data()->set_ciphertext_associated_data(
      "invalid associated data");
  *write_request->mutable_first_request_metadata() = metadata;
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_EQ(response.write().status().code(), grpc::INVALID_ARGUMENT);

  google::rpc::Status finalize_config;
  finalize_config.set_code(grpc::StatusCode::OK);
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      finalize_config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&finalize_response));

  ASSERT_THAT(finalize_response,
              EqualsProto(R"pb(read {
                                 first_response_metadata {
                                   total_size_bytes: 11
                                   compression_type: COMPRESSION_TYPE_NONE
                                   unencrypted {}
                                 }
                                 finish_read: true
                                 data: "test result"
                               })pb"));
}

TEST_F(ConfidentialTransformServerBaseTest, ReadBlobOnConfigure) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure)
      .WillOnce([](ConfigureRequest request, Session::Context& context) {
        context.EmitUnencrypted(Session::KV(
            PARSE_TEXT_PROTO(R"pb(type_url: "xyz")pb"), "abc", "key_id"));
        return ConfigureResponse{};
      });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = stub_->Session(&context);

  SessionRequest configure_request;
  configure_request.mutable_configure()->set_chunk_size(/*chunk_size=*/1000);
  CHECK(stream->Write(configure_request));

  SessionResponse read_response, configure_response;
  CHECK(stream->Read(&read_response));
  CHECK(stream->Read(&configure_response));

  EXPECT_THAT(
      read_response,
      EqualsProto(R"pb(read {
                         first_response_metadata {
                           total_size_bytes: 3
                           compression_type: COMPRESSION_TYPE_NONE
                           unencrypted { blob_id: "key_id" }
                         }
                         finish_read: true
                         data: "abc"
                         first_response_configuration { type_url: "xyz" }
                       })pb"));
}

TEST_F(ConfidentialTransformServerBaseTest, ReadBlobOnWrite) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Write)
      .WillOnce([](WriteRequest request, std::string unencrypted_data,
                   Session::Context& context) {
        context.Emit(PARSE_TEXT_PROTO(
            R"pb(first_response_metadata {
                   total_size_bytes: 3
                   compression_type: COMPRESSION_TYPE_NONE
                   unencrypted {}
                 }
                 first_response_configuration { type_url: "xyz" }
                 data: "abc")pb"));
        return WriteFinishedResponse{};
      });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest write_request = CreateDefaultWriteRequest("foo");
  CHECK(stream->Write(write_request));

  SessionResponse read_response, write_response;
  CHECK(stream->Read(&read_response));
  CHECK(stream->Read(&write_response));

  EXPECT_THAT(
      read_response,
      EqualsProto(R"pb(read {
                         first_response_metadata {
                           total_size_bytes: 3
                           compression_type: COMPRESSION_TYPE_NONE
                           unencrypted {}
                         }
                         first_response_configuration { type_url: "xyz" }
                         finish_read: true
                         data: "abc"
                       })pb"));
}

TEST_F(ConfidentialTransformServerBaseTest, ReadBlobOnCommit) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Commit)
      .WillOnce([](CommitRequest request, Session::Context& context) {
        context.Emit(PARSE_TEXT_PROTO(
            R"pb(first_response_metadata {
                   total_size_bytes: 3
                   compression_type: COMPRESSION_TYPE_NONE
                   unencrypted {}
                 }
                 first_response_configuration { type_url: "xyz" }
                 data: "abc")pb"));
        return CommitResponse{};
      });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest commit_request;
  commit_request.mutable_commit();
  CHECK(stream->Write(commit_request));

  SessionResponse read_response, commit_response;
  CHECK(stream->Read(&read_response));
  CHECK(stream->Read(&commit_response));

  EXPECT_THAT(
      read_response,
      EqualsProto(R"pb(read {
                         first_response_metadata {
                           total_size_bytes: 3
                           compression_type: COMPRESSION_TYPE_NONE
                           unencrypted {}
                         }
                         first_response_configuration { type_url: "xyz" }
                         finish_read: true
                         data: "abc"
                       })pb"));
}

TEST_F(ConfidentialTransformServerBaseTest, ReadMultipleBlobsOnFinalize) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce([](FinalizeRequest request, BlobMetadata unused,
                   Session::Context& context) {
        context.Emit(PARSE_TEXT_PROTO(
            R"pb(first_response_metadata {
                   total_size_bytes: 5
                   compression_type: COMPRESSION_TYPE_NONE
                   unencrypted {}
                 }
                 first_response_configuration { type_url: "abc" }
                 data: "first")pb"));
        context.Emit(PARSE_TEXT_PROTO(
            R"pb(first_response_metadata {
                   total_size_bytes: 6
                   compression_type: COMPRESSION_TYPE_NONE
                   unencrypted {}
                 }
                 first_response_configuration { type_url: "xyz" }
                 data: "second")pb"));
        return FinalizeResponse{};
      });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  CHECK(stream->Write(finalize_request));

  SessionResponse read_response1, read_response2, finalize_response;
  CHECK(stream->Read(&read_response1));
  CHECK(stream->Read(&read_response2));
  CHECK(stream->Read(&finalize_response));

  EXPECT_THAT(
      read_response1,
      EqualsProto(R"pb(read {
                         first_response_metadata {
                           total_size_bytes: 5
                           compression_type: COMPRESSION_TYPE_NONE
                           unencrypted {}
                         }
                         first_response_configuration { type_url: "abc" }
                         finish_read: true
                         data: "first"
                       })pb"));
  EXPECT_THAT(
      read_response2,
      EqualsProto(R"pb(read {
                         first_response_metadata {
                           total_size_bytes: 6
                           compression_type: COMPRESSION_TYPE_NONE
                           unencrypted {}
                         }
                         first_response_configuration { type_url: "xyz" }
                         finish_read: true
                         data: "second"
                       })pb"));
  ASSERT_THAT(finalize_response, EqualsProto(R"pb(finalize {})pb"));
}

TEST_F(ConfidentialTransformServerBaseTest, ReadNoBlobsOnFinalize) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce([](FinalizeRequest request, BlobMetadata unused,
                   Session::Context& context) { return FinalizeResponse{}; });
  service_->AddSession(std::move(mock_session));
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  CHECK(stream->Write(finalize_request));
  SessionResponse finalize_response;
  CHECK(stream->Read(&finalize_response));
  EXPECT_TRUE(finalize_response.has_finalize());
}

TEST_F(ConfidentialTransformServerBaseTest, EmitError) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce(
          [](FinalizeRequest request, BlobMetadata unused,
             Session::Context& context) -> absl::StatusOr<FinalizeResponse> {
            if (!context.EmitError(
                    absl::InternalError("Expected Internal Error"))) {
              LOG(ERROR) << "EmitError failed";
              return absl::InternalError("EmitError failed");
            }
            return FinalizeResponse{};
          });
  service_->AddSession(std::move(mock_session));

  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  SessionResponse error_response, finalize_response;

  CHECK(stream->Write(finalize_request));
  CHECK(stream->Read(&error_response));
  CHECK(stream->Read(&finalize_response));

  // Verify that an error has been emitted.
  EXPECT_TRUE(error_response.has_write());
  EXPECT_EQ(error_response.write().status().code(), grpc::StatusCode::INTERNAL);
  EXPECT_THAT(error_response.write().status().message(),
              HasSubstr("Expected Internal Error"));

  // Verify that FinalizeResponse has been received.
  EXPECT_TRUE(finalize_response.has_finalize());
}

TEST_F(ConfidentialTransformServerBaseTest, EmitEncrypted) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure)
      .WillOnce(
          [](ConfigureRequest request,
             Session::Context& context) -> absl::StatusOr<ConfigureResponse> {
            if (!context.EmitEncrypted(0, "foobar")) {
              LOG(ERROR) << "EmitEncrypted failed";
              return absl::InternalError("EmitEncrypted failed");
            }
            return ConfigureResponse{};
          });
  service_->AddSession(std::move(mock_session));

  SessionRequest configure_request;
  SessionResponse read_response, configure_response;

  configure_request.mutable_configure()->set_chunk_size(1000);
  grpc::ClientContext context;
  auto stream = stub_->Session(&context);

  CHECK(stream->Write(configure_request));
  CHECK(stream->Read(&read_response));
  CHECK(stream->Read(&configure_response));

  // Verify that encrypted blob has been read
  EXPECT_TRUE(read_response.has_read());
  EXPECT_TRUE(
      read_response.read().first_response_metadata().has_hpke_plus_aead_data());
  const auto& hpke_plus_aead_data =
      read_response.read().first_response_metadata().hpke_plus_aead_data();
  EXPECT_GT(hpke_plus_aead_data.ciphertext_associated_data().size(), 0);
  EXPECT_GT(hpke_plus_aead_data.encrypted_symmetric_key().size(), 0);
  EXPECT_GT(hpke_plus_aead_data.kms_symmetric_key_associated_data()
                .record_header()
                .size(),
            0);
  // Random blob ID is present
  EXPECT_GT(hpke_plus_aead_data.blob_id().size(), 0);
}

TEST_F(ConfidentialTransformServerBaseTest, EmitReleasable) {
  InitializeTransform();
  auto mock_session = std::make_unique<MockSession>();
  EXPECT_CALL(*mock_session, Configure).WillOnce(Return(ConfigureResponse{}));
  EXPECT_CALL(*mock_session, Finalize)
      .WillOnce(
          [](FinalizeRequest request, BlobMetadata unused,
             Session::Context& context) -> absl::StatusOr<FinalizeResponse> {
            std::string release_token;
            if (!context.EmitReleasable(0, "foobar", "src", "dst",
                                        release_token)) {
              LOG(ERROR) << "EmitReleasable failed";
              return absl::InternalError("EmitReleasable failed");
            }
            FinalizeResponse response;
            *response.mutable_release_token() = std::move(release_token);
            return response;
          });
  service_->AddSession(std::move(mock_session));

  grpc::ClientContext context;
  auto stream = StartSession(&context);

  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  SessionResponse read_response, finalize_response;

  CHECK(stream->Write(finalize_request));
  CHECK(stream->Read(&read_response));
  CHECK(stream->Read(&finalize_response));

  // Verify that encrypted blob has been read
  EXPECT_TRUE(read_response.has_read());
  EXPECT_TRUE(
      read_response.read().first_response_metadata().has_hpke_plus_aead_data());
  const auto& hpke_plus_aead_data =
      read_response.read().first_response_metadata().hpke_plus_aead_data();
  EXPECT_GT(hpke_plus_aead_data.ciphertext_associated_data().size(), 0);
  EXPECT_GT(hpke_plus_aead_data.encrypted_symmetric_key().size(), 0);
  EXPECT_GT(hpke_plus_aead_data.kms_symmetric_key_associated_data()
                .record_header()
                .size(),
            0);
  // Random blob ID is present
  EXPECT_GT(hpke_plus_aead_data.blob_id().size(), 0);

  // Verify that release token is present
  EXPECT_TRUE(finalize_response.has_finalize());
  EXPECT_GT(finalize_response.finalize().release_token().size(), 0);
}

}  // namespace

}  // namespace confidential_federated_compute
