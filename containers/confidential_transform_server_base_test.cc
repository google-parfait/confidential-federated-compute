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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "containers/session.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
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
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute {

namespace {

using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::NonceAndCounter;
using ::fcp::confidential_compute::NonceGenerator;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigurationMetadata;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::oak::containers::v1::MockOrchestratorCryptoStub;
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::Test;

inline constexpr int kMaxNumSessions = 8;

class MockSession final : public confidential_federated_compute::Session {
 public:
  MOCK_METHOD(absl::Status, ConfigureSession,
              (SessionRequest configure_request), (override));
  MOCK_METHOD(absl::StatusOr<SessionResponse>, SessionWrite,
              (const WriteRequest& write_request, std::string unencrypted_data),
              (override));
  MOCK_METHOD(absl::StatusOr<SessionResponse>, SessionCommit,
              (const CommitRequest& commit_request), (override));
  MOCK_METHOD(absl::StatusOr<SessionResponse>, FinalizeSession,
              (const FinalizeRequest& request,
               const BlobMetadata& input_metadata),
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

SessionResponse GetDefaultFinalizeResponse() {
  SessionResponse response;
  ReadResponse* read_response = response.mutable_read();
  read_response->set_finish_read(true);
  std::string result = "test result";
  *(read_response->mutable_data()) = result;
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  metadata.set_total_size_bytes(result.size());
  *(read_response->mutable_first_response_metadata()) = metadata;
  return response;
}

class FakeConfidentialTransform final
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  FakeConfidentialTransform(
      oak::containers::v1::OrchestratorCrypto::StubInterface* crypto_stub)
      : ConfidentialTransformBase(crypto_stub) {};

  void AddSession(
      std::unique_ptr<confidential_federated_compute::MockSession> session) {
    session_ = std::move(session);
  };

 protected:
  virtual absl::StatusOr<google::protobuf::Struct> InitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) {
    google::rpc::Status config_status;
    if (!request->configuration().UnpackTo(&config_status)) {
      return absl::InvalidArgumentError("Config cannot be unpacked.");
    }
    if (config_status.code() != grpc::StatusCode::OK) {
      return absl::InvalidArgumentError("Invalid config.");
    }
    return google::protobuf::Struct();
  }
  virtual absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) {
    FCP_ASSIGN_OR_RETURN(google::protobuf::Struct config_properties,
                         InitializeTransform(request));
    return config_properties;
  }

  virtual absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) {
    return absl::OkStatus();
  }

  virtual absl::StatusOr<
      std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() {
    if (session_ == nullptr) {
      auto session =
          std::make_unique<confidential_federated_compute::MockSession>();
      EXPECT_CALL(*session, ConfigureSession(_))
          .WillOnce(Return(absl::OkStatus()));
      return std::move(session);
    }
    return std::move(session_);
  }

 private:
  std::unique_ptr<confidential_federated_compute::MockSession> session_;
};

class ConfidentialTransformServerBaseTest : public Test {
 public:
  ConfidentialTransformServerBaseTest() {
    int port;
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    stub_ = ConfidentialTransform::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~ConfidentialTransformServerBaseTest() override { server_->Shutdown(); }

 protected:
  testing::NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub_;
  FakeConfidentialTransform service_{&mock_crypto_stub_};
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
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
  google::rpc::Status config_status;
  config_status.set_code(grpc::StatusCode::OK);
  StreamInitializeRequest initialize_request;
  initialize_request.mutable_initialize_request()
      ->mutable_configuration()
      ->PackFrom(config_status);
  initialize_request.mutable_initialize_request()->set_max_num_sessions(
      kMaxNumSessions);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(write_configuration));
  ASSERT_TRUE(writer->Write(initialize_request));
  // WritesDone is called to indicate that no more messages will be sent.
  ASSERT_TRUE(writer->WritesDone());
  // Finish is called to get the server's response and final status of the
  // stream.
  ASSERT_TRUE(writer->Finish().ok());
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(response.public_key());
  ASSERT_TRUE(cwt.ok());
}

TEST_F(ConfidentialTransformServerBaseTest,
       ValidStreamInitializeNoWriteConfiguration) {
  grpc::ClientContext context;
  InitializeResponse response;

  google::rpc::Status config_status;
  config_status.set_code(grpc::StatusCode::OK);
  StreamInitializeRequest initialize_request;
  initialize_request.mutable_initialize_request()
      ->mutable_configuration()
      ->PackFrom(config_status);
  initialize_request.mutable_initialize_request()->set_max_num_sessions(
      kMaxNumSessions);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(initialize_request));
  ASSERT_TRUE(writer->WritesDone());
  ASSERT_TRUE(writer->Finish().ok());
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(response.public_key());
  ASSERT_TRUE(cwt.ok());
}

TEST_F(ConfidentialTransformServerBaseTest,
       StreamInitializeInitializeRequestBeforeWriteConfiguration) {
  grpc::ClientContext context;
  InitializeResponse response;

  google::rpc::Status config_status;
  config_status.set_code(grpc::StatusCode::OK);
  StreamInitializeRequest initialize_request;
  initialize_request.mutable_initialize_request()
      ->mutable_configuration()
      ->PackFrom(config_status);
  initialize_request.mutable_initialize_request()->set_max_num_sessions(
      kMaxNumSessions);

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
  StreamInitializeRequest request;
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
  google::rpc::Status config_status;
  config_status.set_code(grpc::StatusCode::OK);

  StreamInitializeRequest request;
  request.mutable_initialize_request()->mutable_configuration()->PackFrom(
      config_status);
  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(request));
  ASSERT_TRUE(writer->WritesDone());
  ASSERT_TRUE(writer->Finish().ok());

  grpc::ClientContext second_context;
  request.mutable_initialize_request()->mutable_configuration()->PackFrom(
      config_status);
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

  google::rpc::Status config_status;
  config_status.set_code(grpc::StatusCode::OK);
  StreamInitializeRequest initialize_request;
  initialize_request.mutable_initialize_request()
      ->mutable_configuration()
      ->PackFrom(config_status);

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
  configure_request.mutable_configure()->mutable_configuration();

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_FALSE(stream->Read(&configure_response));
  auto status = stream->Finish();
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("StreamInitialize must be called before Session"));
}

TEST_F(ConfidentialTransformServerBaseTest,
       StreamInitializeSessionConfigureGeneratesNonce) {
  grpc::ClientContext configure_context;
  InitializeResponse response;
  google::rpc::Status config_status;
  config_status.set_code(grpc::StatusCode::OK);
  StreamInitializeRequest initialize_request;
  initialize_request.mutable_initialize_request()
      ->mutable_configuration()
      ->PackFrom(config_status);
  initialize_request.mutable_initialize_request()->set_max_num_sessions(
      kMaxNumSessions);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&configure_context, &response);
  ASSERT_TRUE(writer->Write(initialize_request));
  ASSERT_TRUE(writer->WritesDone());
  ASSERT_TRUE(writer->Finish().ok());

  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_configure();

  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session, FinalizeSession(_, _))
      .WillOnce(Return(GetDefaultFinalizeResponse()));
  service_.AddSession(std::move(mock_session));

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_TRUE(stream->Read(&session_response));

  ASSERT_TRUE(session_response.has_configure());
  ASSERT_GT(session_response.configure().nonce().size(), 0);

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&finalize_response));
  ASSERT_TRUE(stream->Finish().ok());
}

TEST_F(ConfidentialTransformServerBaseTest,
       StreamInitializeSessionRejectsMoreThanMaximumNumSessions) {
  grpc::ClientContext configure_context;
  InitializeResponse response;
  google::rpc::Status config_status;
  config_status.set_code(grpc::StatusCode::OK);
  StreamInitializeRequest initialize_request;
  initialize_request.mutable_initialize_request()
      ->mutable_configuration()
      ->PackFrom(config_status);
  initialize_request.mutable_initialize_request()->set_max_num_sessions(
      kMaxNumSessions);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&configure_context, &response);
  ASSERT_TRUE(writer->Write(initialize_request));
  ASSERT_TRUE(writer->WritesDone());
  ASSERT_TRUE(writer->Finish().ok());

  std::vector<std::unique_ptr<
      ::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>>
      streams;
  std::vector<std::unique_ptr<grpc::ClientContext>> contexts;
  for (int i = 0; i < kMaxNumSessions; i++) {
    std::unique_ptr<grpc::ClientContext> session_context =
        std::make_unique<grpc::ClientContext>();
    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure();

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
  rejected_request.mutable_configure();

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&rejected_context);
  ASSERT_TRUE(stream->Write(rejected_request));
  ASSERT_FALSE(stream->Read(&rejected_response));
  ASSERT_EQ(stream->Finish().error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

class InitializedConfidentialTransformServerBaseTest
    : public ConfidentialTransformServerBaseTest {
 public:
  InitializedConfidentialTransformServerBaseTest() {
    grpc::ClientContext configure_context;
    InitializeResponse response;

    google::rpc::Status config_status;
    config_status.set_code(grpc::StatusCode::OK);
    StreamInitializeRequest request;
    request.mutable_initialize_request()->mutable_configuration()->PackFrom(
        config_status);
    request.mutable_initialize_request()->set_max_num_sessions(kMaxNumSessions);

    std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
        stub_->StreamInitialize(&configure_context, &response);
    CHECK(writer->Write(request));
    CHECK(writer->WritesDone());
    CHECK(writer->Finish().ok());
    public_key_ = response.public_key();
  }

 protected:
  void StartSession() {
    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure();

    stream_ = stub_->Session(&session_context_);
    CHECK(stream_->Write(session_request));
    CHECK(stream_->Read(&session_response));
    nonce_generator_ =
        std::make_unique<NonceGenerator>(session_response.configure().nonce());
  }
  grpc::ClientContext session_context_;
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream_;
  std::unique_ptr<NonceGenerator> nonce_generator_;
  std::string public_key_;
};

TEST_F(InitializedConfidentialTransformServerBaseTest,
       SessionWritesAndFinalizes) {
  std::string data = "test data";
  SessionRequest write_request = CreateDefaultWriteRequest(data);
  SessionResponse write_response;

  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session, SessionWrite(_, _))
      .WillRepeatedly(Return(
          ToSessionWriteFinishedResponse(absl::OkStatus(), data.size())));
  EXPECT_CALL(*mock_session, SessionCommit(_))
      .WillRepeatedly(Return(ToSessionCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, FinalizeSession(_, _))
      .WillOnce(Return(GetDefaultFinalizeResponse()));
  service_.AddSession(std::move(mock_session));
  StartSession();

  // Accumulate the same unencrypted blob twice.
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));
  ASSERT_TRUE(stream_->Finish().ok());

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());
}

TEST_F(InitializedConfidentialTransformServerBaseTest,
       SessionWritesChunkedBlobAndFinalizes) {
  std::string chunk1 = "abcdef";
  std::string chunk2 = "ghijkl";
  std::string chunk3 = "mno";
  std::string data = chunk1 + chunk2 + chunk3;

  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session, SessionWrite(_, data))
      .WillOnce(Return(
          ToSessionWriteFinishedResponse(absl::OkStatus(), data.size())));
  EXPECT_CALL(*mock_session, SessionCommit(_))
      .WillRepeatedly(Return(ToSessionCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, FinalizeSession(_, _))
      .WillOnce(Return(GetDefaultFinalizeResponse()));
  service_.AddSession(std::move(mock_session));
  StartSession();

  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  metadata.set_total_size_bytes(data.size());
  SessionRequest write_request1;
  *write_request1.mutable_write()->mutable_first_request_metadata() = metadata;
  write_request1.mutable_write()->set_data(chunk1);
  ASSERT_TRUE(stream_->Write(write_request1));

  SessionRequest write_request2;
  write_request2.mutable_write()->set_data(chunk2);
  ASSERT_TRUE(stream_->Write(write_request2));

  SessionRequest write_request3;
  write_request3.mutable_write()->set_data(chunk3);
  write_request3.mutable_write()->set_commit(true);
  ASSERT_TRUE(stream_->Write(write_request3));

  SessionResponse write_response;
  ASSERT_TRUE(stream_->Read(&write_response));

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));
  ASSERT_TRUE(stream_->Finish().ok());

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());
}

TEST_F(InitializedConfidentialTransformServerBaseTest,
       SessionIgnoresInvalidInputs) {
  std::string data = "test data";
  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session, SessionWrite(_, _))
      .WillOnce(
          Return(ToSessionWriteFinishedResponse(absl::OkStatus(), data.size())))
      .WillOnce(Return(ToSessionWriteFinishedResponse(
          absl::InvalidArgumentError("Invalid argument"), 0)));
  EXPECT_CALL(*mock_session, SessionCommit(_))
      .WillRepeatedly(Return(ToSessionCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, FinalizeSession(_, _))
      .WillOnce(Return(GetDefaultFinalizeResponse()));
  service_.AddSession(std::move(mock_session));
  StartSession();

  SessionRequest write_request_1 = CreateDefaultWriteRequest(data);
  SessionResponse write_response_1;

  ASSERT_TRUE(stream_->Write(write_request_1));
  ASSERT_TRUE(stream_->Read(&write_response_1));

  SessionRequest write_request_2 = CreateDefaultWriteRequest(data);
  SessionResponse write_response_2;

  ASSERT_TRUE(stream_->Write(write_request_2));
  ASSERT_TRUE(stream_->Read(&write_response_2));

  ASSERT_TRUE(write_response_2.has_write());
  ASSERT_EQ(write_response_2.write().committed_size_bytes(), 0);
  ASSERT_EQ(write_response_2.write().status().code(), grpc::INVALID_ARGUMENT);

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  ASSERT_TRUE(finalize_response.read().finish_read());
  ASSERT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  ASSERT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());
}

TEST_F(InitializedConfidentialTransformServerBaseTest,
       SessionFailsIfWriteFails) {
  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session, SessionWrite(_, _))
      .WillOnce(Return(absl::InternalError("Internal Error")));
  service_.AddSession(std::move(mock_session));
  StartSession();

  SessionRequest write_request_1 = CreateDefaultWriteRequest("test data");
  SessionResponse write_response_1;

  ASSERT_TRUE(stream_->Write(write_request_1));
  ASSERT_FALSE(stream_->Read(&write_response_1));
  ASSERT_EQ(stream_->Finish().error_code(), grpc::StatusCode::INTERNAL);
}

TEST_F(InitializedConfidentialTransformServerBaseTest,
       SessionFailsIfIncompleteChunkedBlobWriteFollowedByWrite) {
  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  service_.AddSession(std::move(mock_session));
  StartSession();

  SessionRequest write_request_1 = CreateDefaultWriteRequest("test data");
  SessionRequest write_request_2 = CreateDefaultWriteRequest("test data");
  SessionResponse write_response;

  write_request_1.mutable_write()->set_commit(false);
  ASSERT_TRUE(stream_->Write(write_request_1));
  ASSERT_TRUE(stream_->Write(write_request_2));
  ASSERT_FALSE(stream_->Read(&write_response));
  ASSERT_EQ(stream_->Finish().error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(InitializedConfidentialTransformServerBaseTest,
       SessionFailsIfIncompleteChunkedBlobWriteFollowedByFinalize) {
  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  service_.AddSession(std::move(mock_session));
  StartSession();

  SessionRequest write_request = CreateDefaultWriteRequest("test data");
  SessionRequest finalize_request;
  SessionResponse finalize_response;

  write_request.mutable_write()->set_commit(false);
  ASSERT_TRUE(stream_->Write(write_request));

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_FALSE(stream_->Read(&finalize_response));
  ASSERT_EQ(stream_->Finish().error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(InitializedConfidentialTransformServerBaseTest,
       SessionFailsIfCommitFails) {
  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session, SessionCommit(_))
      .WillOnce(Return(absl::InternalError("Internal Error")));
  service_.AddSession(std::move(mock_session));
  StartSession();

  SessionRequest commit_request;
  commit_request.mutable_commit();
  SessionResponse commit_response;

  ASSERT_TRUE(stream_->Write(commit_request));
  ASSERT_FALSE(stream_->Read(&commit_response));
  ASSERT_EQ(stream_->Finish().error_code(), grpc::StatusCode::INTERNAL);
}

TEST_F(InitializedConfidentialTransformServerBaseTest,
       SessionFailsIfFinalizeFails) {
  std::string data = "test data";
  SessionRequest write_request = CreateDefaultWriteRequest(data);
  SessionResponse write_response;

  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session, SessionWrite(_, _))
      .WillRepeatedly(Return(
          ToSessionWriteFinishedResponse(absl::OkStatus(), data.size())));

  EXPECT_CALL(*mock_session, SessionCommit(_))
      .WillRepeatedly(Return(ToSessionCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, FinalizeSession(_, _))
      .WillOnce(Return(absl::InternalError("Internal Error")));
  service_.AddSession(std::move(mock_session));
  StartSession();

  // Accumulate the same unencrypted blob twice.
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));
  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  google::rpc::Status config;
  config.set_code(grpc::StatusCode::INTERNAL);
  SessionRequest finalize_request;
  SessionResponse finalize_response;
  finalize_request.mutable_finalize()->mutable_configuration()->PackFrom(
      config);
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_FALSE(stream_->Read(&finalize_response));
  ASSERT_EQ(stream_->Finish().error_code(), grpc::StatusCode::INTERNAL);
}

TEST_F(InitializedConfidentialTransformServerBaseTest,
       SessionDecryptsMultipleRecords) {
  std::string message_0 = "test data 0";
  std::string message_1 = "test data 1";
  std::string message_2 = "test data 2";

  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session, SessionWrite(_, _))
      .WillOnce(Return(
          ToSessionWriteFinishedResponse(absl::OkStatus(), message_0.size())))
      .WillOnce(Return(
          ToSessionWriteFinishedResponse(absl::OkStatus(), message_1.size())))
      .WillOnce(Return(
          ToSessionWriteFinishedResponse(absl::OkStatus(), message_2.size())));
  EXPECT_CALL(*mock_session, SessionCommit(_))
      .WillRepeatedly(Return(ToSessionCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, FinalizeSession(_, _))
      .WillOnce(Return(GetDefaultFinalizeResponse()));
  service_.AddSession(std::move(mock_session));
  StartSession();

  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());
  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();

  absl::StatusOr<NonceAndCounter> nonce_0 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_TRUE(nonce_0.ok());
  absl::StatusOr<Record> rewrapped_record_0 =
      crypto_test_utils::CreateRewrappedRecord(
          message_0, ciphertext_associated_data, public_key_,
          nonce_0->blob_nonce, *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_0.ok()) << rewrapped_record_0.status();

  SessionRequest request_0;
  WriteRequest* write_request_0 = request_0.mutable_write();
  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  *write_request_0->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_0);
  write_request_0->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_0->counter);
  write_request_0->mutable_first_request_configuration()->PackFrom(config);
  write_request_0->set_commit(true);
  write_request_0->set_data(
      rewrapped_record_0->hpke_plus_aead_data().ciphertext());

  SessionResponse response_0;

  ASSERT_TRUE(stream_->Write(request_0));
  ASSERT_TRUE(stream_->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), grpc::OK);

  absl::StatusOr<NonceAndCounter> nonce_1 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_TRUE(nonce_1.ok());
  absl::StatusOr<Record> rewrapped_record_1 =
      crypto_test_utils::CreateRewrappedRecord(
          message_1, ciphertext_associated_data, public_key_,
          nonce_1->blob_nonce, *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_1.ok()) << rewrapped_record_1.status();

  SessionRequest request_1;
  WriteRequest* write_request_1 = request_1.mutable_write();
  *write_request_1->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_1);
  write_request_1->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_1->counter);
  write_request_1->mutable_first_request_configuration()->PackFrom(config);
  write_request_1->set_commit(true);
  write_request_1->set_data(
      rewrapped_record_1->hpke_plus_aead_data().ciphertext());

  SessionResponse response_1;

  ASSERT_TRUE(stream_->Write(request_1));
  ASSERT_TRUE(stream_->Read(&response_1));
  ASSERT_EQ(response_1.write().status().code(), grpc::OK);

  absl::StatusOr<NonceAndCounter> nonce_2 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_TRUE(nonce_2.ok());
  absl::StatusOr<Record> rewrapped_record_2 =
      crypto_test_utils::CreateRewrappedRecord(
          message_2, ciphertext_associated_data, public_key_,
          nonce_2->blob_nonce, *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_2.ok()) << rewrapped_record_2.status();

  SessionRequest request_2;
  WriteRequest* write_request_2 = request_2.mutable_write();
  *write_request_2->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_2);
  write_request_2->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_2->counter);
  write_request_2->mutable_first_request_configuration()->PackFrom(config);
  write_request_2->set_commit(true);
  write_request_2->set_data(
      rewrapped_record_2->hpke_plus_aead_data().ciphertext());

  SessionResponse response_2;

  ASSERT_TRUE(stream_->Write(request_2));
  ASSERT_TRUE(stream_->Read(&response_2));
  ASSERT_EQ(response_2.write().status().code(), grpc::OK);

  google::rpc::Status finalize_config;
  finalize_config.set_code(grpc::StatusCode::OK);
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
}

TEST_F(InitializedConfidentialTransformServerBaseTest,
       TransformIgnoresUndecryptableInputs) {
  std::string message_1 = "test data 1";

  auto mock_session =
      std::make_unique<confidential_federated_compute::MockSession>();
  EXPECT_CALL(*mock_session, ConfigureSession(_))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session, SessionWrite(_, _))
      .WillOnce(Return(
          ToSessionWriteFinishedResponse(absl::OkStatus(), message_1.size())));
  EXPECT_CALL(*mock_session, SessionCommit(_))
      .WillRepeatedly(Return(ToSessionCommitResponse(absl::OkStatus())));
  EXPECT_CALL(*mock_session, FinalizeSession(_, _))
      .WillOnce(Return(GetDefaultFinalizeResponse()));
  service_.AddSession(std::move(mock_session));
  StartSession();

  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());
  std::string ciphertext_associated_data = "ciphertext associated data";

  // Create one record that will fail to decrypt and one record that can be
  // successfully decrypted.
  std::string message_0 = "test data 0";
  absl::StatusOr<NonceAndCounter> nonce_0 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_TRUE(nonce_0.ok());
  absl::StatusOr<Record> rewrapped_record_0 =
      crypto_test_utils::CreateRewrappedRecord(
          message_0, ciphertext_associated_data, public_key_,
          nonce_0->blob_nonce, *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_0.ok()) << rewrapped_record_0.status();

  SessionRequest request_0;
  WriteRequest* write_request_0 = request_0.mutable_write();
  google::rpc::Status config;
  config.set_code(grpc::StatusCode::OK);
  *write_request_0->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_0);
  write_request_0->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_0->counter);
  write_request_0->mutable_first_request_configuration()->PackFrom(config);
  write_request_0->set_commit(true);
  write_request_0->set_data("undecryptable");

  SessionResponse response_0;

  ASSERT_TRUE(stream_->Write(request_0));
  ASSERT_TRUE(stream_->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), grpc::INVALID_ARGUMENT);

  absl::StatusOr<NonceAndCounter> nonce_1 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_TRUE(nonce_1.ok());
  absl::StatusOr<Record> rewrapped_record_1 =
      crypto_test_utils::CreateRewrappedRecord(
          message_1, ciphertext_associated_data, public_key_,
          nonce_1->blob_nonce, *reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_1.ok()) << rewrapped_record_1.status();

  SessionRequest request_1;
  WriteRequest* write_request_1 = request_1.mutable_write();
  *write_request_1->mutable_first_request_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record_1);
  write_request_1->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_1->counter);
  write_request_1->mutable_first_request_configuration()->PackFrom(config);
  write_request_1->set_commit(true);
  write_request_1->set_data(
      rewrapped_record_1->hpke_plus_aead_data().ciphertext());

  SessionResponse response_1;

  ASSERT_TRUE(stream_->Write(request_1));
  ASSERT_TRUE(stream_->Read(&response_1));
  ASSERT_EQ(response_1.write().status().code(), grpc::OK);

  google::rpc::Status finalize_config;
  finalize_config.set_code(grpc::StatusCode::OK);
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
}

}  // namespace

}  // namespace confidential_federated_compute
