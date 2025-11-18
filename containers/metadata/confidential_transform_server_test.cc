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
#include "containers/metadata/confidential_transform_server.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "cc/crypto/client_encryptor.h"
#include "cc/crypto/encryption_key.h"
#include "containers/crypto_test_utils.h"
#include "fcp/base/status_converters.h"
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
using ::fcp::base::FromGrpcStatus;
using ::fcp::confidentialcompute::AuthorizeConfidentialTransformResponse;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::MetadataContainerConfig;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::google::protobuf::Value;
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

class MetadataConfidentialTransformTest : public Test {
 protected:
  MetadataConfidentialTransformTest() {
    int port;
    const std::string server_address = "[::1]:";

    auto encryption_key_handle = std::make_unique<EncryptionKeyProvider>(
        EncryptionKeyProvider::Create().value());
    oak_client_encryptor_ =
        ClientEncryptor::Create(encryption_key_handle->GetSerializedPublicKey())
            .value();
    service_ = std::make_unique<MetadataConfidentialTransform>(
        std::make_unique<NiceMock<MockSigningKeyHandle>>(),
        std::move(encryption_key_handle));

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    stub_ = ConfidentialTransform::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~MetadataConfidentialTransformTest() override { server_->Shutdown(); }

  std::unique_ptr<MetadataConfidentialTransform> service_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
  std::unique_ptr<ClientEncryptor> oak_client_encryptor_;
};

TEST_F(MetadataConfidentialTransformTest, StreamInitializeTransformFails) {
  ClientContext context;
  InitializeRequest request;
  InitializeResponse response;

  EXPECT_THAT(
      WriteInitializeRequest(stub_->StreamInitialize(&context, &response),
                             std::move(request)),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Metadata container must be initialized with KMS")));
}

TEST_F(MetadataConfidentialTransformTest, ReadWriteConfigurationRequestFails) {
  ClientContext context;
  InitializeResponse response;
  StreamInitializeRequest write_config_request;
  write_config_request.mutable_write_configuration()->set_commit(true);

  std::unique_ptr<ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(write_config_request));

  InitializeRequest init_request;
  EXPECT_THAT(
      WriteInitializeRequest(std::move(writer), std::move(init_request)),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Metadata container does not support "
                         "WriteConfigurationRequests")));
}

TEST_F(MetadataConfidentialTransformTest, StreamInitializeWithKmsSucceeds) {
  ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
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
  request.mutable_configuration()->PackFrom(config);

  MetadataContainerConfig config_constraints;
  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(config_constraints);
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;
  auto writer = stub_->StreamInitialize(&context, &response);

  EXPECT_THAT(WriteInitializeRequest(std::move(writer), std::move(request)),
              IsOk());
}

TEST_F(MetadataConfidentialTransformTest,
       StreamInitializeWithKmsInvalidConfiguration) {
  ClientContext context;
  InitializeRequest request;
  InitializeResponse response;
  Value value;
  request.mutable_configuration()->PackFrom(value);

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
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

}  // namespace
}  // namespace confidential_federated_compute::metadata
