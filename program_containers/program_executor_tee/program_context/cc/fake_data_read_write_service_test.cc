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

#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/client_context.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/sync_stream.h"
#include "gtest/gtest.h"
#include "openssl/rand.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

using ::confidential_federated_compute::crypto_test_utils::MockSigningKeyHandle;
using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::outgoing::DataReadWrite;
using ::fcp::confidentialcompute::outgoing::ReadRequest;
using ::fcp::confidentialcompute::outgoing::ReadResponse;
using ::fcp::confidentialcompute::outgoing::WriteRequest;
using ::fcp::confidentialcompute::outgoing::WriteResponse;
using ::grpc::ClientContext;
using ::grpc::ClientReader;
using ::grpc::ClientWriter;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::PrintToStringParamName;
using ::testing::Test;

class FakeDataReadWriteServiceTest : public ::testing::TestWithParam<bool> {
 public:
  static std::string TestNameSuffix(
      const ::testing::TestParamInfo<bool>& info) {
    return info.param ? "WithKms" : "NoKms";
  }

  void SetUp() override {
    fake_data_read_write_service_ =
        std::make_unique<FakeDataReadWriteService>(GetParam());
    google::protobuf::Struct config_properties;
    input_blob_decryptor_ =
        std::make_unique<confidential_federated_compute::BlobDecryptor>(
            mock_signing_key_handle_, config_properties,
            std::vector<absl::string_view>(
                {fake_data_read_write_service_->GetInputPublicPrivateKeyPair()
                     .second}));

    const std::string server_address = "[::1]:";
    int port;
    ServerBuilder data_read_write_builder;
    data_read_write_builder.AddListeningPort(absl::StrCat(server_address, 0),
                                             grpc::InsecureServerCredentials(),
                                             &port);
    data_read_write_builder.RegisterService(
        fake_data_read_write_service_.get());
    fake_data_read_write_server_ = data_read_write_builder.BuildAndStart();
    LOG(INFO) << "DataReadWrite server listening on "
              << server_address + std::to_string(port) << std::endl;

    stub_ = DataReadWrite::NewStub(
        grpc::CreateChannel(absl::StrCat(server_address, port),
                            grpc::InsecureChannelCredentials()));
  }

  absl::Status CreateWriteRequest(WriteRequest* write_request, std::string key,
                                  std::string data) {
    FCP_ASSIGN_OR_RETURN(OkpKey okp_key,
                         OkpKey::Decode(fake_data_read_write_service_
                                            ->GetResultPublicPrivateKeyPair()
                                            .second));
    BlobHeader header;
    std::string blob_id(kBlobIdSize, '\0');
    (void)RAND_bytes(reinterpret_cast<unsigned char*>(blob_id.data()),
                     blob_id.size());
    header.set_blob_id(blob_id);
    header.set_key_id(okp_key.key_id);
    header.set_access_policy_sha256(kAccessPolicyHash);
    std::string serialized_blob_header = header.SerializeAsString();

    MessageEncryptor message_encryptor;
    FCP_ASSIGN_OR_RETURN(
        EncryptMessageResult encrypted_message,
        message_encryptor.EncryptForRelease(
            data,
            fake_data_read_write_service_->GetResultPublicPrivateKeyPair()
                .first,
            serialized_blob_header,
            /*src_state=*/"", /*dst_state=*/"",
            [this](absl::string_view message) -> absl::StatusOr<std::string> {
              FCP_ASSIGN_OR_RETURN(auto signature,
                                   mock_signing_key_handle_.Sign(message));
              return std::move(*signature.mutable_signature());
            }));

    BlobMetadata metadata;
    metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
    metadata.set_total_size_bytes(encrypted_message.ciphertext.size());
    BlobMetadata::HpkePlusAeadMetadata* hpke_plus_aead_metadata =
        metadata.mutable_hpke_plus_aead_data();
    hpke_plus_aead_metadata->set_ciphertext_associated_data(
        std::string(serialized_blob_header));
    hpke_plus_aead_metadata->set_encrypted_symmetric_key(
        encrypted_message.encrypted_symmetric_key);
    hpke_plus_aead_metadata->set_encapsulated_public_key(
        encrypted_message.encapped_key);
    hpke_plus_aead_metadata->mutable_kms_symmetric_key_associated_data()
        ->set_record_header(std::string(serialized_blob_header));

    *write_request->mutable_first_request_metadata() = std::move(metadata);
    write_request->set_commit(true);
    write_request->set_data(encrypted_message.ciphertext);
    write_request->set_release_token(encrypted_message.release_token);
    write_request->set_key(key);

    return absl::OkStatus();
  }

  void TearDown() override { fake_data_read_write_server_->Shutdown(); }

 protected:
  NiceMock<MockSigningKeyHandle> mock_signing_key_handle_;
  std::unique_ptr<BlobDecryptor> input_blob_decryptor_;

  std::unique_ptr<FakeDataReadWriteService> fake_data_read_write_service_;
  std::unique_ptr<Server> fake_data_read_write_server_;
  std::unique_ptr<DataReadWrite::Stub> stub_;
};

TEST_P(FakeDataReadWriteServiceTest, ReadRequestSuccessForEncryptedMessage) {
  std::string uri = "uri";
  std::string message = "message";
  if (GetParam()) {
    // Prepopulate the FakeDataReadWriteService with the encrypted data.
    CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForKms(
        uri, message));

    // Make a ReadRequest for the uri.
    ReadRequest read_request;
    read_request.set_uri(uri);
    ClientContext client_context;
    std::unique_ptr<ClientReader<ReadResponse>> reader(
        stub_->Read(&client_context, read_request));

    // Read the response and check that decrypted message matches.
    ReadResponse read_response;
    ASSERT_TRUE(reader->Read(&read_response));
    ASSERT_TRUE(read_response.has_first_response_metadata());
    BlobHeader blob_header;
    EXPECT_TRUE(
        blob_header.ParseFromString(read_response.first_response_metadata()
                                        .hpke_plus_aead_data()
                                        .kms_symmetric_key_associated_data()
                                        .record_header()));
    EXPECT_THAT(input_blob_decryptor_->DecryptBlob(
                    read_response.first_response_metadata(),
                    read_response.data(), blob_header.key_id()),
                message);

  } else {
    // Prepopulate the FakeDataReadWriteService with the encrypted data.
    std::string nonce = "nonce";
    absl::StatusOr<absl::string_view> recipient_public_key =
        input_blob_decryptor_->GetPublicKey();
    ASSERT_TRUE(recipient_public_key.ok());
    CHECK_OK(fake_data_read_write_service_->StoreEncryptedMessageForLedger(
        uri, message, "ciphertext associated data", *recipient_public_key,
        nonce, "reencryption_public_key"));

    // Make a ReadRequest for the uri.
    ReadRequest read_request;
    read_request.set_uri(uri);
    read_request.set_nonce(nonce);
    ClientContext client_context;
    std::unique_ptr<ClientReader<ReadResponse>> reader(
        stub_->Read(&client_context, read_request));

    // Read the response and check that the nonce and decrypted message match.
    ReadResponse read_response;
    ASSERT_TRUE(reader->Read(&read_response));
    ASSERT_TRUE(read_response.has_first_response_metadata());
    EXPECT_EQ(read_response.first_response_metadata()
                  .hpke_plus_aead_data()
                  .rewrapped_symmetric_key_associated_data()
                  .nonce(),
              nonce);
    EXPECT_THAT(
        input_blob_decryptor_->DecryptBlob(
            read_response.first_response_metadata(), read_response.data()),
        message);
  }

  // Check that the DataReadWrite service logged the uri.
  std::vector<std::string> requested_uris =
      fake_data_read_write_service_->GetReadRequestUris();
  EXPECT_EQ(requested_uris.size(), 1);
  EXPECT_EQ(requested_uris[0], uri);
}

TEST_P(FakeDataReadWriteServiceTest, ReadRequestSuccessForPlaintextMessage) {
  // Prepopulate the FakeDataReadWriteService with the plaintext data.
  std::string uri = "uri";
  std::string message = "message";
  CHECK_OK(fake_data_read_write_service_->StorePlaintextMessage(uri, message));

  // Make a ReadRequest for the uri.
  ReadRequest read_request;
  read_request.set_uri(std::string(uri));
  ClientContext client_context;
  std::unique_ptr<ClientReader<ReadResponse>> reader(
      stub_->Read(&client_context, read_request));

  // Read the response and check that received message matches.
  ReadResponse read_response;
  ASSERT_TRUE(reader->Read(&read_response));
  ASSERT_TRUE(read_response.has_first_response_metadata());
  EXPECT_EQ(read_response.first_response_metadata().encryption_metadata_case(),
            BlobMetadata::EncryptionMetadataCase::kUnencrypted);
  EXPECT_EQ(read_response.data(), message);

  // Check that the DataReadWrite service logged the uri.
  std::vector<std::string> requested_uris =
      fake_data_read_write_service_->GetReadRequestUris();
  EXPECT_EQ(requested_uris.size(), 1);
  EXPECT_EQ(requested_uris[0], uri);
}

TEST_P(FakeDataReadWriteServiceTest, ReadRequestSuccessForMultipleMessages) {
  // Prepopulate the FakeDataReadWriteService with some data.
  std::vector<std::string> storage_uris = {"uri_1", "uri_2"};
  std::string base_message = "message";
  for (std::string& uri : storage_uris) {
    std::string message = absl::StrCat(uri, base_message);
    CHECK_OK(
        fake_data_read_write_service_->StorePlaintextMessage(uri, message));
  }

  // Attempt to resolve a sequence of uris that correspond to the stored data.
  std::vector<std::string> read_uris = {"uri_2", "uri_1", "uri_1"};
  for (std::string& uri : read_uris) {
    // Make a ReadRequest for the uri.
    ReadRequest read_request;
    read_request.set_uri(std::string(uri));
    ClientContext client_context;
    std::unique_ptr<ClientReader<ReadResponse>> reader(
        stub_->Read(&client_context, read_request));

    // Read the response and check that the received message matches.
    ReadResponse read_response;
    ASSERT_TRUE(reader->Read(&read_response));
    ASSERT_TRUE(read_response.has_first_response_metadata());
    EXPECT_EQ(
        read_response.first_response_metadata().encryption_metadata_case(),
        BlobMetadata::EncryptionMetadataCase::kUnencrypted);
    EXPECT_EQ(read_response.data(), absl::StrCat(uri, base_message));
  }

  // Check that the DataReadWrite service logged the correct sequence of uris.
  std::vector<std::string> requested_uris =
      fake_data_read_write_service_->GetReadRequestUris();
  EXPECT_EQ(requested_uris.size(), 3);
  EXPECT_EQ(requested_uris[0], read_uris[0]);
  EXPECT_EQ(requested_uris[1], read_uris[1]);
  EXPECT_EQ(requested_uris[2], read_uris[2]);
}

TEST_P(FakeDataReadWriteServiceTest, ReadRequestFailureForUnknownUri) {
  // Prepopulate the FakeDataReadWriteService with the plaintext data.
  std::string message = "message";
  CHECK_OK(
      fake_data_read_write_service_->StorePlaintextMessage("uri", message));

  // Make a ReadRequest for a different uri.
  ReadRequest read_request;
  read_request.set_uri("other_uri");
  ClientContext client_context;
  std::unique_ptr<ClientReader<ReadResponse>> reader(
      stub_->Read(&client_context, read_request));

  // Try to read the response
  grpc::Status status = reader->Finish();
  ASSERT_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);
  ASSERT_THAT(status.error_message(),
              HasSubstr("Requested uri other_uri not found"));
}

TEST_P(FakeDataReadWriteServiceTest, ReadRequestFailureForAlreadySetUri) {
  // Populate the FakeDataReadWriteService with a plaintext message.
  std::string uri = "uri";
  std::string message = "message";
  CHECK_OK(fake_data_read_write_service_->StorePlaintextMessage(uri, message));

  // Check that adding a plaintext message with the same uri fails.
  auto store_result =
      fake_data_read_write_service_->StorePlaintextMessage(uri, message);
  EXPECT_EQ(store_result, absl::InvalidArgumentError("Uri already set."));

  // Check that adding an encrypted message with the same uri fails.
  if (GetParam()) {
    store_result = fake_data_read_write_service_->StoreEncryptedMessageForKms(
        uri, message);
  } else {
    std::string nonce = "nonce";
    absl::StatusOr<absl::string_view> recipient_public_key =
        input_blob_decryptor_->GetPublicKey();
    ASSERT_TRUE(recipient_public_key.ok());
    store_result =
        fake_data_read_write_service_->StoreEncryptedMessageForLedger(
            uri, message, "ciphertext associated data", *recipient_public_key,
            nonce, "reencryption_public_key");
  }
  EXPECT_EQ(store_result, absl::InvalidArgumentError("Uri already set."));
}

TEST_P(FakeDataReadWriteServiceTest, WriteRequestSuccess) {
  WriteRequest write_request_1;
  WriteRequest write_request_2;

  if (GetParam()) {
    ASSERT_TRUE(
        CreateWriteRequest(&write_request_1, "key_1", "write_request_1").ok());
    ASSERT_TRUE(
        CreateWriteRequest(&write_request_2, "key_2", "write_request_2").ok());
  } else {
    write_request_1.mutable_first_request_metadata()
        ->mutable_unencrypted()
        ->set_blob_id("key_1");
    write_request_1.set_data("write_request_1");
    write_request_2.mutable_first_request_metadata()
        ->mutable_unencrypted()
        ->set_blob_id("key_2");
    write_request_2.set_data("write_request_2");
  }

  for (const auto& write_request : {write_request_1, write_request_2}) {
    ClientContext client_context;
    WriteResponse response;
    std::unique_ptr<ClientWriter<WriteRequest>> stream(
        stub_->Write(&client_context, &response));
    stream->Write(write_request);
    stream->WritesDone();
    ASSERT_TRUE(stream->Finish().ok());
  }

  // Check that the WriteRequests were parsed and recorded correctly.
  std::map<std::string, std::string> released_data =
      fake_data_read_write_service_->GetReleasedData();
  ASSERT_EQ(released_data.size(), 2);
  ASSERT_EQ(released_data["key_1"], "write_request_1");
  ASSERT_EQ(released_data["key_2"], "write_request_2");
}

INSTANTIATE_TEST_SUITE_P(KmsParam, FakeDataReadWriteServiceTest,
                         ::testing::Bool(),  // Generates {false, true}
                         FakeDataReadWriteServiceTest::TestNameSuffix);

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee