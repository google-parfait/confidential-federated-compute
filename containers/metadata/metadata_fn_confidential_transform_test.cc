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
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "cc/crypto/client_encryptor.h"
#include "cc/crypto/encryption_key.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "containers/fns/confidential_transform_server.h"
#include "containers/metadata/metadata_map_fn.h"
#include "containers/metadata/testing/test_utils.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/kms.pb.h"
#include "fcp/protos/confidentialcompute/tee_payload_metadata.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gtest/gtest.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::metadata {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::crypto_test_utils::MockSigningKeyHandle;
using ::confidential_federated_compute::fns::FnConfidentialTransform;
using ::confidential_federated_compute::metadata::testing::
    BuildEncryptedCheckpoint;
using ::confidential_federated_compute::metadata::testing::EqualsEventTimeRange;
using ::confidential_federated_compute::metadata::testing::LowerNBitsAreZero;
using ::fcp::base::FromGrpcStatus;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::MetadataContainerConfig;
using ::fcp::confidentialcompute::PayloadMetadataSet;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::TeePayloadMetadata;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::ClientContext;
using ::grpc::ClientWriter;
using ::oak::crypto::ClientEncryptor;
using ::oak::crypto::EncryptionKeyProvider;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Test;

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

class MetadataConfidentialTransformInitializeTest : public Test {
 protected:
  MetadataConfidentialTransformInitializeTest() {
    int port;
    const std::string server_address = "[::1]:";

    std::tie(public_key_, private_key_) =
        crypto_test_utils::GenerateKeyPair(key_id_);
    auto encryption_key_handle = std::make_unique<EncryptionKeyProvider>(
        EncryptionKeyProvider::Create().value());
    oak_client_encryptor_ =
        ClientEncryptor::Create(encryption_key_handle->GetSerializedPublicKey())
            .value();
    service_ = std::make_unique<FnConfidentialTransform>(
        std::make_unique<NiceMock<MockSigningKeyHandle>>(),
        ProvideMetadataMapFnFactory, std::move(encryption_key_handle));

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    stub_ = ConfidentialTransform::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~MetadataConfidentialTransformInitializeTest() override {
    server_->Shutdown();
  }

  std::string key_id_ = "key_id";
  std::string allowed_policy_hash_ = "hash_1";
  std::string public_key_;
  std::string private_key_;
  std::unique_ptr<FnConfidentialTransform> service_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
  std::unique_ptr<ClientEncryptor> oak_client_encryptor_;
};

TEST_F(MetadataConfidentialTransformInitializeTest,
       StreamInitializeWithKmsSucceeds) {
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(
      MetadataContainerConfig());
  associated_data.add_authorized_logical_pipeline_policies_hashes(
      allowed_policy_hash_);

  ClientContext context;
  InitializeRequest request;
  request.set_max_num_sessions(1);
  InitializeResponse response;

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  protected_response.add_decryption_keys(private_key_);

  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;
  auto writer = stub_->StreamInitialize(&context, &response);

  ASSERT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());
}

TEST_F(MetadataConfidentialTransformInitializeTest,
       StreamInitializeWithKmsInvalidConfiguration) {
  ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  PayloadMetadataSet value;
  request.mutable_configuration()->PackFrom(value);

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.add_authorized_logical_pipeline_policies_hashes(
      allowed_policy_hash_);
  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;
  auto writer = stub_->StreamInitialize(&context, &response);

  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Config constraints cannot be unpacked")));
}

class MetadataConfidentialTransformSessionTest
    : public MetadataConfidentialTransformInitializeTest {
 protected:
  MetadataConfidentialTransformSessionTest()
      : MetadataConfidentialTransformInitializeTest() {}

  void SetUp() override {
    MetadataContainerConfig config = PARSE_TEXT_PROTO(
        R"pb(
          metadata_configs {
            key: "test_config"
            value {
              num_partitions: 10
              event_time_range_granularity: EVENT_TIME_GRANULARITY_DAY
            }
          }
        )pb");
    AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
    associated_data.add_authorized_logical_pipeline_policies_hashes(
        allowed_policy_hash_);
    associated_data.mutable_config_constraints()->PackFrom(config);
    ClientContext context;
    InitializeRequest request;
    request.set_max_num_sessions(1);
    InitializeResponse response;

    AuthorizeConfidentialTransformResponse::ProtectedResponse
        protected_response;
    *protected_response.add_result_encryption_keys() = "result_encryption_key";
    protected_response.add_decryption_keys(private_key_);

    auto encrypted_request =
        oak_client_encryptor_
            ->Encrypt(protected_response.SerializeAsString(),
                      associated_data.SerializeAsString())
            .value();
    *request.mutable_protected_response() = encrypted_request;
    auto writer = stub_->StreamInitialize(&context, &response);

    ASSERT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
                IsOk());
  }
};

TEST_F(MetadataConfidentialTransformSessionTest, CreateSessionSucceeds) {
  // Create a session and configure it.
  grpc::ClientContext session_context;
  SessionRequest configure_request;
  SessionResponse configure_response;
  configure_request.mutable_configure()->set_chunk_size(1000);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(configure_request));
  ASSERT_TRUE(stream->Read(&configure_response));
  EXPECT_TRUE(configure_response.has_configure());

  // Write a checkpoint to the session.
  BlobHeader header;
  header.set_blob_id("blob_id");
  header.set_key_id(key_id_);
  header.set_access_policy_sha256(allowed_policy_hash_);
  std::pair<BlobMetadata, std::string> checkpoint = BuildEncryptedCheckpoint(
      "privacy_id", {"2025-01-01T12:00:00+00:00", "2025-01-02T12:00:00+00:00"},
      public_key_, header.SerializeAsString());
  SessionRequest write_request;
  WriteRequest* write = write_request.mutable_write();
  write->set_commit(true);
  write->set_data(checkpoint.second);
  *write->mutable_first_request_metadata() = checkpoint.first;
  ASSERT_TRUE(stream->Write(write_request));
  SessionResponse read_response;
  ASSERT_TRUE(stream->Read(&read_response));
  SessionResponse write_finished_response;
  ASSERT_TRUE(stream->Read(&write_finished_response));
  ASSERT_TRUE(write_finished_response.has_write());
  ASSERT_TRUE(read_response.has_read());

  // The extracted payload metadata should be in the configuration of the read
  // response.
  PayloadMetadataSet metadata_set;
  read_response.read().first_response_configuration().UnpackTo(&metadata_set);
  EXPECT_EQ(metadata_set.metadata_size(), 1);
  ASSERT_TRUE(metadata_set.metadata().contains("test_config"));

  TeePayloadMetadata tee_metadata = metadata_set.metadata().at("test_config");
  // Verify partition key.
  EXPECT_THAT(tee_metadata.partition_key(), LowerNBitsAreZero(60));
  // Verify event time range.
  EXPECT_THAT(tee_metadata.event_time_range(),
              EqualsEventTimeRange(2025, 1, 1, 2025, 1, 3));

  // Go through the rest of the session flow to ensure that it can be completed
  // successfully.
  SessionRequest finalize_request;
  finalize_request.mutable_finalize();
  ASSERT_TRUE(stream->Write(finalize_request));
  SessionResponse finalize_response;
  ASSERT_TRUE(stream->Read(&finalize_response));
  ASSERT_TRUE(finalize_response.has_finalize());
  ASSERT_TRUE(stream->WritesDone());
  ASSERT_THAT(FromGrpcStatus(stream->Finish()), IsOk());
}

}  // namespace
}  // namespace confidential_federated_compute::metadata
