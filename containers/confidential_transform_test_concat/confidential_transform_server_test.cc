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
#include "containers/confidential_transform_test_concat/confidential_transform_server.h"

#include <memory>
#include <string>
#include <tuple>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "cc/crypto/client_encryptor.h"
#include "cc/crypto/encryption_key.h"
#include "containers/crypto_test_utils.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::confidential_transform_test_concat {

namespace {

using ::confidential_federated_compute::crypto_test_utils::MockSigningKeyHandle;
using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::ClientWriter;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::StatusCode;
using ::oak::crypto::ClientEncryptor;
using ::oak::crypto::EncryptionKeyProvider;
using ::testing::NiceMock;
using ::testing::Test;

class TestConcatServerTest : public Test {
 public:
  TestConcatServerTest() {
    int port;
    const std::string server_address = "[::1]:";
    auto encryption_key_handle = std::make_unique<EncryptionKeyProvider>(
        EncryptionKeyProvider::Create().value());
    oak_client_encryptor_ =
        ClientEncryptor::Create(encryption_key_handle->GetSerializedPublicKey())
            .value();
    service_ = std::make_unique<TestConcatConfidentialTransform>(
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

  ~TestConcatServerTest() override { server_->Shutdown(); }

 protected:
  void InitializeTransform() {
    grpc::ClientContext context;
    InitializeRequest request;
    InitializeResponse response;
    request.set_max_num_sessions(8);

    auto public_private_key_pair = crypto_test_utils::GenerateKeyPair(key_id_);
    public_key_ = public_private_key_pair.first;

    AuthorizeConfidentialTransformResponse::ProtectedResponse
        protected_response;
    protected_response.add_result_encryption_keys(public_key_);
    protected_response.add_decryption_keys(public_private_key_pair.second);
    AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
    associated_data.add_authorized_logical_pipeline_policies_hashes("hash");
    auto encrypted_request =
        oak_client_encryptor_
            ->Encrypt(protected_response.SerializeAsString(),
                      associated_data.SerializeAsString())
            .value();
    *request.mutable_protected_response() = encrypted_request;

    CHECK(TryWriteInitializeRequest(
        std::move(request), stub_->StreamInitialize(&context, &response)));
  }

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
  StartSession(grpc::ClientContext* context) {
    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure()->set_chunk_size(1000);

    std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
        stream;
    stream = stub_->Session(context);
    CHECK(stream->Write(session_request));
    CHECK(stream->Read(&session_response));
    return stream;
  }

  SessionRequest CreateUnencryptedWriteRequest(std::string data) {
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

  std::unique_ptr<ConfidentialTransform::Stub> stub_;
  std::string key_id_ = "key_id";

 private:
  // Attempt to write the InitializeRequest to the client stream and then close
  // the stream, returning the result of Finish.
  bool TryWriteInitializeRequest(
      InitializeRequest request,
      std::unique_ptr<ClientWriter<StreamInitializeRequest>> init_stream) {
    StreamInitializeRequest stream_request;
    *stream_request.mutable_initialize_request() = std::move(request);
    return init_stream->Write(stream_request) && init_stream->WritesDone() &&
           init_stream->Finish().ok();
  }

  std::unique_ptr<TestConcatConfidentialTransform> service_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<ClientEncryptor> oak_client_encryptor_;
  std::string public_key_;
};

TEST_F(TestConcatServerTest, SessionBeforeInitialize) {
  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->mutable_configuration();

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_FALSE(stream->Read(&configure_response));
  EXPECT_EQ(stream->Finish().error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(TestConcatServerTest, SessionWriteCommitFinalizeUnencryptedBlob) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = StartSession(&context);
  std::string data1 = "data1";
  SessionRequest write_request = CreateUnencryptedWriteRequest(data1);
  SessionResponse write_response;
  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));
  ASSERT_TRUE(write_response.has_write());
  EXPECT_EQ(write_response.write().committed_size_bytes(), data1.size());
  EXPECT_EQ(write_response.write().status().code(), grpc::OK);

  std::string data2 = "data2";
  write_request = CreateUnencryptedWriteRequest(data2);
  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));
  ASSERT_TRUE(write_response.has_write());
  EXPECT_EQ(write_response.write().committed_size_bytes(), data1.size());
  EXPECT_EQ(write_response.write().status().code(), grpc::OK);

  SessionRequest commit_request;
  commit_request.mutable_commit();
  SessionResponse commit_response;
  ASSERT_TRUE(stream->Write(commit_request));
  ASSERT_TRUE(stream->Read(&commit_response));
  EXPECT_EQ(commit_response.commit().status().code(), grpc::OK);

  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  SessionResponse finalize_response;
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&finalize_response));
  ASSERT_TRUE(finalize_response.has_read());
  EXPECT_TRUE(finalize_response.read().finish_read());
  EXPECT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  EXPECT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());
  EXPECT_EQ(finalize_response.read().data(), "data1data2");
}

TEST_F(TestConcatServerTest, SessionIgnoresUndecryptableInputs) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = StartSession(&context);
  std::string message = "data";
  BlobHeader header;
  header.set_blob_id("blob_id");
  header.set_key_id(key_id_);
  auto [metadata, ciphertext] = Encrypt(message, header.SerializeAsString());

  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->set_commit(true);
  write_request->set_data(ciphertext);

  SessionResponse response;
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_EQ(response.write().status().code(), grpc::OK);
  ASSERT_EQ(response.write().committed_size_bytes(), ciphertext.size());

  // Try to write an invalid message.
  metadata.mutable_hpke_plus_aead_data()->set_ciphertext_associated_data(
      "invalid associated data");
  *write_request->mutable_first_request_metadata() = metadata;
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_EQ(response.write().status().code(), grpc::INVALID_ARGUMENT);

  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  SessionResponse finalize_response;
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  EXPECT_TRUE(finalize_response.read().finish_read());
  EXPECT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  EXPECT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());
  EXPECT_EQ(finalize_response.read().data(), message);
}

TEST_F(TestConcatServerTest, SessionWriteCommitFinalizeEncryptedBlob) {
  InitializeTransform();
  grpc::ClientContext context;
  auto stream = StartSession(&context);
  std::string message = "data";
  BlobHeader header;
  header.set_blob_id("blob_id");
  header.set_key_id(key_id_);
  auto [metadata, ciphertext] = Encrypt(message, header.SerializeAsString());

  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->set_commit(true);
  write_request->set_data(ciphertext);

  SessionResponse response;
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_EQ(response.write().status().code(), grpc::OK);
  ASSERT_EQ(response.write().committed_size_bytes(), ciphertext.size());

  SessionRequest commit_request;
  commit_request.mutable_commit();
  SessionResponse commit_response;
  ASSERT_TRUE(stream->Write(commit_request));
  ASSERT_TRUE(stream->Read(&commit_response));
  EXPECT_EQ(commit_response.commit().status().code(), grpc::OK);

  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  SessionResponse finalize_response;
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&finalize_response));
  ASSERT_TRUE(finalize_response.has_read());
  EXPECT_TRUE(finalize_response.read().finish_read());
  EXPECT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  EXPECT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());
  EXPECT_EQ(finalize_response.read().data(), "data");
}

}  // namespace

}  // namespace
   // confidential_federated_compute::confidential_transform_test_concat
