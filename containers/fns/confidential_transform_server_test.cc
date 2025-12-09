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

#include "containers/fns/confidential_transform_server.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "cc/crypto/client_encryptor.h"
#include "cc/crypto/encryption_key.h"
#include "containers/crypto_test_utils.h"
#include "containers/fns/fn.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gtest/gtest.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fns {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::crypto_test_utils::MockSigningKeyHandle;
using ::fcp::base::FromGrpcStatus;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::google::protobuf::Any;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::crypto::ClientEncryptor;
using ::oak::crypto::EncryptionKeyProvider;
using ::oak::crypto::v1::EncryptedRequest;
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;

// Write the InitializeRequest to the client stream and then close
// the stream, returning the status of Finish.
absl::Status WriteInitializeRequest(
    std::unique_ptr<grpc::ClientWriter<StreamInitializeRequest>> stream,
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

class MockFn : public Fn {
 public:
  MOCK_METHOD((absl::Status), InitializeReplica,
              (google::protobuf::Any config, Context& context), (override));
  MOCK_METHOD((absl::Status), FinalizeReplica,
              (google::protobuf::Any config, Context& context), (override));
  MOCK_METHOD((absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse>),
              Write,
              (fcp::confidentialcompute::WriteRequest request,
               std::string unencrypted_data, Context& context),
              (override));
  MOCK_METHOD((absl::StatusOr<fcp::confidentialcompute::CommitResponse>),
              Commit,
              (fcp::confidentialcompute::CommitRequest request,
               Context& context),
              (override));
};

class MockFnFactory : public FnFactory {
 public:
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Fn>>, CreateFn, (),
              (const, override));
};

class MockFnFactoryProvider {
 public:
  MOCK_METHOD((absl::StatusOr<std::unique_ptr<FnFactory>>), Create,
              (const Any&, const Any&));
};

class FnConfidentialTransformTest : public ::testing::Test {
 protected:
  FnConfidentialTransformTest() {
    int port;
    const std::string server_address = "[::1]:";

    auto encryption_key_handle = std::make_unique<EncryptionKeyProvider>(
        EncryptionKeyProvider::Create().value());
    oak_client_encryptor_ =
        ClientEncryptor::Create(encryption_key_handle->GetSerializedPublicKey())
            .value();
    service_ = std::make_unique<FnConfidentialTransform>(
        std::make_unique<NiceMock<MockSigningKeyHandle>>(),
        [this](const Any& config, const Any& config_constraints) {
          return mock_fn_factory_provider_.Create(config, config_constraints);
        },
        std::move(encryption_key_handle));

    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on "
              << server_address + std::to_string(port);
    stub_ = ConfidentialTransform::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~FnConfidentialTransformTest() override { server_->Shutdown(); }

  NiceMock<MockFnFactoryProvider> mock_fn_factory_provider_;
  std::unique_ptr<FnConfidentialTransform> service_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
  std::unique_ptr<ClientEncryptor> oak_client_encryptor_;
};

// Create an encrypted ProtectedResponse with the given result encryption key
// and config constraints.
absl::StatusOr<EncryptedRequest> CreateEncryptedProtectedResponse(
    std::string result_encryption_key, ClientEncryptor& oak_client_encryptor,
    Any config_constraints = Any()) {
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  *associated_data.mutable_config_constraints() = config_constraints;
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  protected_response.add_result_encryption_keys(result_encryption_key);

  return oak_client_encryptor.Encrypt(protected_response.SerializeAsString(),
                                      associated_data.SerializeAsString());
}

TEST_F(FnConfidentialTransformTest, InitializeWithKmsSucceeds) {
  EXPECT_CALL(mock_fn_factory_provider_, Create(_, _))
      .WillOnce(Return(std::make_unique<NiceMock<MockFnFactory>>()));

  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;

  absl::StatusOr<EncryptedRequest> encrypted_request =
      CreateEncryptedProtectedResponse("result_encryption_key",
                                       *oak_client_encryptor_);
  ASSERT_THAT(encrypted_request, IsOk());
  *request.mutable_protected_response() = *encrypted_request;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());
}

TEST_F(FnConfidentialTransformTest, WriteConfigurationRequestFails) {
  grpc::ClientContext context;
  InitializeResponse response;
  StreamInitializeRequest write_configuration;
  write_configuration.mutable_write_configuration()->set_commit(true);

  auto writer = stub_->StreamInitialize(&context, &response);

  ASSERT_TRUE(writer->Write(write_configuration));
  EXPECT_THAT(
      FromGrpcStatus(writer->Finish()),
      StatusIs(
          absl::StatusCode::kUnimplemented,
          HasSubstr(
              "Fn container does not support WriteConfigurationRequests yet")));
}

TEST_F(FnConfidentialTransformTest, FnFactoryCreationFails) {
  EXPECT_CALL(mock_fn_factory_provider_, Create(_, _))
      .WillOnce(
          Return(absl::InternalError("This FnFactory cannot be created.")));

  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;

  absl::StatusOr<EncryptedRequest> encrypted_request =
      CreateEncryptedProtectedResponse("result_encryption_key",
                                       *oak_client_encryptor_);
  ASSERT_THAT(encrypted_request, IsOk());
  *request.mutable_protected_response() = *encrypted_request;

  auto writer = stub_->StreamInitialize(&context, &response);

  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("This FnFactory cannot be created.")));
}

TEST_F(FnConfidentialTransformTest, NonKmsInitializeFails) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(FnConfidentialTransformTest, AlreadyInitializedFails) {
  // Expect the FnFactory to be created once. The second attempted
  // initialization should fail before calling the FnFactory factory again.
  EXPECT_CALL(mock_fn_factory_provider_, Create(_, _))
      .WillOnce(Return(std::make_unique<NiceMock<MockFnFactory>>()));
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;

  absl::StatusOr<EncryptedRequest> encrypted_request =
      CreateEncryptedProtectedResponse("result_encryption_key",
                                       *oak_client_encryptor_);
  ASSERT_THAT(encrypted_request, IsOk());

  *request.mutable_protected_response() = *encrypted_request;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());
  // Second initialization should fail.
  grpc::ClientContext second_context;
  InitializeRequest second_request;
  *second_request.mutable_protected_response() = *encrypted_request;
  InitializeResponse second_response;
  auto second_writer =
      stub_->StreamInitialize(&second_context, &second_response);
  EXPECT_THAT(WriteInitializeRequest(std::move(second_writer),
                                     std::move(second_request)),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Fn container already initialized.")));
}

TEST_F(FnConfidentialTransformTest, SessionCallsCreateFn) {
  auto mock_fn = std::make_unique<MockFn>();
  // Expect the Configure method to be called once and return an OK status.
  EXPECT_CALL(*mock_fn, InitializeReplica).WillOnce(Return(absl::OkStatus()));

  // Expect that the FnFactory is called to create a Fn exactly once
  auto mock_fn_factory = std::make_unique<NiceMock<MockFnFactory>>();
  EXPECT_CALL(*mock_fn_factory, CreateFn).WillOnce(Return(std::move(mock_fn)));
  EXPECT_CALL(mock_fn_factory_provider_, Create)
      .WillOnce(Return(std::move(mock_fn_factory)));

  grpc::ClientContext context;
  InitializeRequest request;
  request.set_max_num_sessions(1);
  InitializeResponse response;

  absl::StatusOr<EncryptedRequest> encrypted_request =
      CreateEncryptedProtectedResponse("result_encryption_key",
                                       *oak_client_encryptor_);
  ASSERT_THAT(encrypted_request, IsOk());
  *request.mutable_protected_response() = *encrypted_request;

  auto writer = stub_->StreamInitialize(&context, &response);
  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());

  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_configure()->set_chunk_size(1000);
  session_request.mutable_configure()->mutable_configuration();

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_TRUE(stream->Read(&session_response));
  ASSERT_TRUE(stream->WritesDone());
}

TEST_F(FnConfidentialTransformTest, CreateSessionNotInitialized) {
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

}  // namespace
}  // namespace confidential_federated_compute::fns
