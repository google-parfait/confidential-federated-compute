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

using ::absl_testing::IsOk;
using ::confidential_federated_compute::crypto_test_utils::MockSigningKeyHandle;
using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::NonceAndCounter;
using ::fcp::confidential_compute::NonceGenerator;
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
  std::unique_ptr<TestConcatConfidentialTransform> service_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
  std::unique_ptr<ClientEncryptor> oak_client_encryptor_;
};

TEST_F(TestConcatServerTest, ValidStreamInitialize) {
  grpc::ClientContext context;
  InitializeResponse response;
  InitializeRequest request;
  request.set_max_num_sessions(8);

  ASSERT_TRUE(TryWriteInitializeRequest(
      std::move(request), stub_->StreamInitialize(&context, &response)));
}

TEST_F(TestConcatServerTest, SessionConfigureGeneratesNonce) {
  grpc::ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  request.set_max_num_sessions(8);

  ASSERT_TRUE(TryWriteInitializeRequest(
      std::move(request), stub_->StreamInitialize(&context, &response)));

  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_configure()->set_chunk_size(1000);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_TRUE(stream->Read(&session_response));

  ASSERT_TRUE(session_response.has_configure());
  EXPECT_GT(session_response.configure().nonce().size(), 0);
}

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

class TestConcatServerSessionTest : public TestConcatServerTest {
 public:
  TestConcatServerSessionTest() {
    grpc::ClientContext context;
    InitializeRequest request;
    InitializeResponse response;
    request.set_max_num_sessions(8);

    CHECK(TryWriteInitializeRequest(
        std::move(request), stub_->StreamInitialize(&context, &response)));
    public_key_ = response.public_key();

    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure()->set_chunk_size(1000);

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

TEST_F(TestConcatServerSessionTest, SessionWritesAndCommitsUnencryptedBlob) {
  std::string data = "data";
  SessionRequest write_request = CreateUnencryptedWriteRequest(data);
  SessionResponse write_response;

  ASSERT_TRUE(stream_->Write(write_request));
  ASSERT_TRUE(stream_->Read(&write_response));

  ASSERT_TRUE(write_response.has_write());
  EXPECT_EQ(write_response.write().committed_size_bytes(), data.size());
  EXPECT_EQ(write_response.write().status().code(), grpc::OK);
}

TEST_F(TestConcatServerSessionTest, SessionWritesAndFinalizesUnencryptedBlobs) {
  std::string data_1 = "one";
  SessionRequest write_request_1 = CreateUnencryptedWriteRequest(data_1);
  SessionResponse write_response_1;

  ASSERT_TRUE(stream_->Write(write_request_1));
  ASSERT_TRUE(stream_->Read(&write_response_1));

  std::string data_2 = "two";
  SessionRequest write_request_2 = CreateUnencryptedWriteRequest(data_2);
  SessionResponse write_response_2;

  ASSERT_TRUE(stream_->Write(write_request_2));
  ASSERT_TRUE(stream_->Read(&write_response_2));

  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  SessionResponse finalize_response;

  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  EXPECT_TRUE(finalize_response.read().finish_read());
  EXPECT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());
  EXPECT_EQ(finalize_response.read().data(), "onetwo");
}

TEST_F(TestConcatServerSessionTest, SessionDecryptsMultipleBlobsAndFinalizes) {
  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_THAT(reencryption_public_key, IsOk());
  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();

  std::string message_0 = "one";
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
  *write_request_0->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_0);
  write_request_0->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_0->counter);
  write_request_0->set_commit(true);
  write_request_0->set_data(std::get<1>(*rewrapped_blob_0));

  SessionResponse response_0;

  ASSERT_TRUE(stream_->Write(request_0));
  ASSERT_TRUE(stream_->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), grpc::OK);

  std::string message_1 = "two";
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
  write_request_1->set_commit(true);
  write_request_1->set_data(std::get<1>(*rewrapped_blob_1));

  SessionResponse response_1;

  ASSERT_TRUE(stream_->Write(request_1));
  ASSERT_TRUE(stream_->Read(&response_1));
  ASSERT_EQ(response_1.write().status().code(), grpc::OK);

  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  SessionResponse finalize_response;
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  EXPECT_TRUE(finalize_response.read().finish_read());
  EXPECT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  EXPECT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());
  EXPECT_EQ(finalize_response.read().data(), "onetwo");
}

TEST_F(TestConcatServerSessionTest, SessionIgnoresUndecryptableInputs) {
  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_THAT(reencryption_public_key, IsOk());
  std::string ciphertext_associated_data =
      BlobHeader::default_instance().SerializeAsString();

  std::string message_0 = "zero";
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
  *write_request_0->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_0);
  write_request_0->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_0->counter);
  write_request_0->set_commit(true);
  write_request_0->set_data(std::get<1>(*rewrapped_blob_0));

  SessionResponse response_0;

  ASSERT_TRUE(stream_->Write(request_0));
  ASSERT_TRUE(stream_->Read(&response_0));
  ASSERT_EQ(response_0.write().status().code(), grpc::OK);
  absl::StatusOr<NonceAndCounter> nonce_1 =
      nonce_generator_->GetNextBlobNonce();
  ASSERT_THAT(nonce_1, IsOk());
  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob_1 =
      crypto_test_utils::CreateRewrappedBlob(
          "unused message", ciphertext_associated_data, public_key_,
          nonce_1->blob_nonce, *reencryption_public_key);
  ASSERT_THAT(rewrapped_blob_1, IsOk());

  SessionRequest invalid_request;
  WriteRequest* invalid_write_request = invalid_request.mutable_write();
  *invalid_write_request->mutable_first_request_metadata() =
      std::get<0>(*rewrapped_blob_1);
  invalid_write_request->mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_counter(nonce_1->counter);
  invalid_write_request->set_commit(true);
  invalid_write_request->set_data("invalid message");

  SessionResponse response_1;

  ASSERT_TRUE(stream_->Write(invalid_request));
  ASSERT_TRUE(stream_->Read(&response_1));
  ASSERT_EQ(response_1.write().status().code(), grpc::INVALID_ARGUMENT);

  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  SessionResponse finalize_response;
  ASSERT_TRUE(stream_->Write(finalize_request));
  ASSERT_TRUE(stream_->Read(&finalize_response));

  ASSERT_TRUE(finalize_response.has_read());
  EXPECT_TRUE(finalize_response.read().finish_read());
  EXPECT_GT(
      finalize_response.read().first_response_metadata().total_size_bytes(), 0);
  EXPECT_TRUE(
      finalize_response.read().first_response_metadata().has_unencrypted());
  EXPECT_EQ(finalize_response.read().data(), message_0);
}

class TestConcatServerSessionKmsTest : public TestConcatServerTest {
 public:
  TestConcatServerSessionKmsTest() {
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

    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure()->set_chunk_size(1000);

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
  std::string public_key_;
};

TEST_F(TestConcatServerSessionKmsTest, SessionWriteRequestSuccess) {
  std::string message = "data";
  BlobHeader header;
  header.set_blob_id("blob_id");
  header.set_key_id(key_id_);
  auto [metadata, ciphertext] =
      EncryptWithKmsKeys(message, header.SerializeAsString());

  SessionRequest request;
  WriteRequest* write_request = request.mutable_write();
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->set_commit(true);
  write_request->set_data(ciphertext);

  SessionResponse response;
  ASSERT_TRUE(stream_->Write(request));
  ASSERT_TRUE(stream_->Read(&response));
  ASSERT_EQ(response.write().status().code(), grpc::OK);
  ASSERT_EQ(response.write().committed_size_bytes(), ciphertext.size());
}

}  // namespace

}  // namespace
   // confidential_federated_compute::confidential_transform_test_concat
