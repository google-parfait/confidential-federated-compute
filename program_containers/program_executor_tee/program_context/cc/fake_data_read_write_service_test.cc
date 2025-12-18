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
#include "program_executor_tee/program_context/cc/kms_helper.h"

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
using ::testing::Test;

class FakeDataReadWriteServiceTest : public ::testing::Test {
 public:
  void SetUp() override {
    fake_data_read_write_service_ =
        std::make_unique<FakeDataReadWriteService>();
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

  void TearDown() override { fake_data_read_write_server_->Shutdown(); }

 protected:
  NiceMock<MockSigningKeyHandle> mock_signing_key_handle_;
  std::unique_ptr<BlobDecryptor> input_blob_decryptor_;

  std::unique_ptr<FakeDataReadWriteService> fake_data_read_write_service_;
  std::unique_ptr<Server> fake_data_read_write_server_;
  std::unique_ptr<DataReadWrite::Stub> stub_;
};

TEST_F(FakeDataReadWriteServiceTest, ReadRequestSuccessForEncryptedMessage) {
  // Prepopulate the FakeDataReadWriteService with the encrypted data.
  std::string uri = "uri";
  std::string message = "message";
  CHECK_OK(
      fake_data_read_write_service_->StoreEncryptedMessageForKms(uri, message));

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
                  read_response.first_response_metadata(), read_response.data(),
                  blob_header.key_id()),
              message);

  // Check that the DataReadWrite service logged the uri.
  std::vector<std::string> requested_uris =
      fake_data_read_write_service_->GetReadRequestUris();
  EXPECT_EQ(requested_uris.size(), 1);
  EXPECT_EQ(requested_uris[0], uri);
}

TEST_F(FakeDataReadWriteServiceTest, ReadRequestSuccessForPlaintextMessage) {
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

TEST_F(FakeDataReadWriteServiceTest, ReadRequestSuccessForMultipleMessages) {
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

TEST_F(FakeDataReadWriteServiceTest, ReadRequestFailureForUnknownUri) {
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

TEST_F(FakeDataReadWriteServiceTest, ReadRequestFailureForAlreadySetUri) {
  // Populate the FakeDataReadWriteService with a plaintext message.
  std::string uri = "uri";
  std::string message = "message";
  CHECK_OK(fake_data_read_write_service_->StorePlaintextMessage(uri, message));

  // Check that adding a plaintext message with the same uri fails.
  auto store_result =
      fake_data_read_write_service_->StorePlaintextMessage(uri, message);
  EXPECT_EQ(store_result, absl::InvalidArgumentError("Uri already set."));

  // Check that adding an encrypted message with the same uri fails.
  store_result =
      fake_data_read_write_service_->StoreEncryptedMessageForKms(uri, message);
  EXPECT_EQ(store_result, absl::InvalidArgumentError("Uri already set."));
}

TEST_F(FakeDataReadWriteServiceTest, WriteRequestSuccess) {
  WriteRequest write_request_1;
  WriteRequest write_request_2;
  auto result_public_key =
      fake_data_read_write_service_->GetResultPublicPrivateKeyPair().first;
  ASSERT_TRUE(CreateWriteRequestForRelease(
                  &write_request_1, mock_signing_key_handle_, result_public_key,
                  "key_1", "write_request_1", kAccessPolicyHash, "state_a",
                  "state_b")
                  .ok());
  ASSERT_TRUE(CreateWriteRequestForRelease(
                  &write_request_2, mock_signing_key_handle_, result_public_key,
                  "key_2", "write_request_2", kAccessPolicyHash, "state_b",
                  "state_c")
                  .ok());

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

  // Check the recorded state transitions as well.
  std::map<std::string, std::pair<std::optional<std::optional<std::string>>,
                                  std::optional<std::string>>>
      released_state_changes =
          fake_data_read_write_service_->GetReleasedStateChanges();
  ASSERT_EQ(released_state_changes.size(), 2);
  auto state_change_1 = released_state_changes["key_1"];
  ASSERT_EQ(state_change_1.first.value().value(), "state_a");
  ASSERT_EQ(state_change_1.second.value(), "state_b");
  auto state_change_2 = released_state_changes["key_2"];
  ASSERT_EQ(state_change_2.first.value().value(), "state_b");
  ASSERT_EQ(state_change_2.second.value(), "state_c");
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee