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

namespace confidential_federated_compute::program_executor_tee {
namespace {

using ::confidential_federated_compute::crypto_test_utils::MockSigningKeyHandle;
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
using ::testing::Test;

class FakeDataReadWriteServiceTest : public Test {
 public:
  FakeDataReadWriteServiceTest() {
    const std::string server_address = "[::1]:";

    ServerBuilder data_read_write_builder;
    data_read_write_builder.AddListeningPort(absl::StrCat(server_address, 0),
                                             grpc::InsecureServerCredentials(),
                                             &port_);
    data_read_write_builder.RegisterService(&fake_data_read_write_service_);
    fake_data_read_write_server_ = data_read_write_builder.BuildAndStart();
    LOG(INFO) << "DataReadWrite server listening on "
              << server_address + std::to_string(port_) << std::endl;

    stub_ = DataReadWrite::NewStub(
        grpc::CreateChannel(absl::StrCat(server_address, port_),
                            grpc::InsecureChannelCredentials()));
  }

  ~FakeDataReadWriteServiceTest() override {
    fake_data_read_write_server_->Shutdown();
  }

 protected:
  int port_;
  FakeDataReadWriteService fake_data_read_write_service_;
  std::unique_ptr<Server> fake_data_read_write_server_;
  std::unique_ptr<DataReadWrite::Stub> stub_;
};

TEST_F(FakeDataReadWriteServiceTest, ReadRequestSuccessForEncryptedMessage) {
  // Prepopulate the FakeDataReadWriteService with the encrypted data.
  std::string uri = "uri";
  std::string message = "message";
  std::string nonce = "nonce";
  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  CHECK_OK(fake_data_read_write_service_.StoreEncryptedMessage(
      uri, message, "ciphertext associated data", *recipient_public_key, nonce,
      "reencryption_public_key"));

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
      blob_decryptor.DecryptBlob(read_response.first_response_metadata(),
                                 read_response.data()),
      message);

  // Check that the DataReadWrite service logged the uri.
  std::vector<std::string> requested_uris =
      fake_data_read_write_service_.GetReadRequestUris();
  EXPECT_EQ(requested_uris.size(), 1);
  EXPECT_EQ(requested_uris[0], uri);
}

TEST_F(FakeDataReadWriteServiceTest, ReadRequestSuccessForPlaintextMessage) {
  // Prepopulate the FakeDataReadWriteService with the plaintext data.
  std::string uri = "uri";
  std::string message = "message";
  CHECK_OK(fake_data_read_write_service_.StorePlaintextMessage(uri, message));

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
      fake_data_read_write_service_.GetReadRequestUris();
  EXPECT_EQ(requested_uris.size(), 1);
  EXPECT_EQ(requested_uris[0], uri);
}

TEST_F(FakeDataReadWriteServiceTest, ReadRequestSuccessForMultipleMessages) {
  // Prepopulate the FakeDataReadWriteService with some data.
  std::vector<std::string> storage_uris = {"uri_1", "uri_2"};
  std::string base_message = "message";
  for (std::string& uri : storage_uris) {
    std::string message = absl::StrCat(uri, base_message);
    CHECK_OK(fake_data_read_write_service_.StorePlaintextMessage(uri, message));
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
      fake_data_read_write_service_.GetReadRequestUris();
  EXPECT_EQ(requested_uris.size(), 3);
  EXPECT_EQ(requested_uris[0], read_uris[0]);
  EXPECT_EQ(requested_uris[1], read_uris[1]);
  EXPECT_EQ(requested_uris[2], read_uris[2]);
}

TEST_F(FakeDataReadWriteServiceTest, ReadRequestFailureForUnknownUri) {
  // Prepopulate the FakeDataReadWriteService with the plaintext data.
  std::string message = "message";
  CHECK_OK(fake_data_read_write_service_.StorePlaintextMessage("uri", message));

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

TEST_F(FakeDataReadWriteServiceTest, ReadRequestFailureForAlreadySetUri) {
  // Populate the FakeDataReadWriteService with a plaintext message.
  std::string uri = "uri";
  std::string message = "message";
  CHECK_OK(fake_data_read_write_service_.StorePlaintextMessage(uri, message));

  // Check that adding a plaintext message with the same uri fails.
  auto store_result =
      fake_data_read_write_service_.StorePlaintextMessage(uri, message);
  EXPECT_EQ(store_result, absl::InvalidArgumentError("Uri already set."));

  // Check that adding an encrypted message with the same uri fails.
  std::string nonce = "nonce";
  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok());
  store_result = fake_data_read_write_service_.StoreEncryptedMessage(
      uri, message, "ciphertext associated data", *recipient_public_key, nonce,
      "reencryption_public_key");
  EXPECT_EQ(store_result, absl::InvalidArgumentError("Uri already set."));
}

TEST_F(FakeDataReadWriteServiceTest, WriteRequestSuccess) {
  // Prepare 5 WriteRequest messages.
  std::vector<WriteRequest> write_requests;
  for (int i = 0; i < 5; i++) {
    WriteRequest write_request;
    write_request.set_data(absl::StrCat("write_request_", i));
    write_requests.push_back(std::move(write_request));
  }

  // Send the first 3 WriteRequests as part of one Write call.
  ClientContext client_context;
  WriteResponse response;
  std::unique_ptr<ClientWriter<WriteRequest>> stream(
      stub_->Write(&client_context, &response));
  stream->Write(write_requests[0]);
  stream->Write(write_requests[1]);
  stream->Write(write_requests[2]);
  stream->WritesDone();
  grpc::Status status = stream->Finish();
  ASSERT_TRUE(status.ok());

  // Send the last 2 WriteRequests as part of another Write call.
  ClientContext another_client_context;
  std::unique_ptr<ClientWriter<WriteRequest>> another_stream(
      stub_->Write(&another_client_context, &response));
  another_stream->Write(write_requests[3]);
  another_stream->Write(write_requests[4]);
  another_stream->WritesDone();
  grpc::Status another_status = another_stream->Finish();
  ASSERT_TRUE(another_status.ok());

  // Check that the WriteRequests were recorded correctly.
  std::vector<std::vector<WriteRequest>> received_write_requests =
      fake_data_read_write_service_.GetWriteCallArgs();
  ASSERT_EQ(received_write_requests.size(), 2);
  ASSERT_EQ(received_write_requests[0].size(), 3);
  ASSERT_EQ(received_write_requests[0][0].data(), "write_request_0");
  ASSERT_EQ(received_write_requests[0][1].data(), "write_request_1");
  ASSERT_EQ(received_write_requests[0][2].data(), "write_request_2");
  ASSERT_EQ(received_write_requests[1].size(), 2);
  ASSERT_EQ(received_write_requests[1][0].data(), "write_request_3");
  ASSERT_EQ(received_write_requests[1][1].data(), "write_request_4");
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee