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
#include "containers/test_concat/pipeline_transform_server.h"

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "proto/containers/orchestrator_crypto_mock.grpc.pb.h"

namespace confidential_federated_compute::test_concat {

namespace {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::GenerateNoncesRequest;
using ::fcp::confidentialcompute::GenerateNoncesResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::StatusCode;
using ::oak::containers::v1::MockOrchestratorCryptoStub;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::Test;

class TestConcatPipelineTransformTest : public Test {
 protected:
  TestConcatPipelineTransformTest() {
    int port;
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    stub_ = PipelineTransform::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~TestConcatPipelineTransformTest() override { server_->Shutdown(); }

  testing::NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub_;
  TestConcatPipelineTransform service_{&mock_crypto_stub_};
  std::unique_ptr<Server> server_;
  std::unique_ptr<PipelineTransform::Stub> stub_;
};

TEST_F(TestConcatPipelineTransformTest, ConfigureAndAttestMoreThanOnce) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;

  ASSERT_TRUE(stub_->ConfigureAndAttest(&context, request, &response).ok());

  grpc::ClientContext second_context;
  auto status = stub_->ConfigureAndAttest(&second_context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("ConfigureAndAttest can only be called once"));
}

TEST_F(TestConcatPipelineTransformTest, ValidConfigureAndAttest) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_TRUE(status.ok());
  ASSERT_THAT(response.public_key(), Not(IsEmpty()));
}

TEST_F(TestConcatPipelineTransformTest, TransformConcatenates) {
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;

  ASSERT_TRUE(stub_
                  ->ConfigureAndAttest(&configure_context, configure_request,
                                       &configure_response)
                  .ok());

  TransformRequest transform_request;
  transform_request.add_inputs()->set_unencrypted_data("1");
  transform_request.add_inputs()->set_unencrypted_data("2");
  transform_request.add_inputs()->set_unencrypted_data("3");
  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  auto transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok());
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());
  ASSERT_EQ(transform_response.outputs(0).unencrypted_data(), "123");
}

TEST_F(TestConcatPipelineTransformTest,
       TransformDecryptsMultipleRecordsAndConcatenates) {
  grpc::ClientContext configure_context;
  ConfigureAndAttestRequest configure_request;
  ConfigureAndAttestResponse configure_response;
  grpc::Status configure_and_attest_status = stub_->ConfigureAndAttest(
      &configure_context, configure_request, &configure_response);
  ASSERT_TRUE(configure_and_attest_status.ok())
      << "ConfigureAndAttest status code was: "
      << configure_and_attest_status.error_code();

  std::string recipient_public_key = configure_response.public_key();

  grpc::ClientContext nonces_context;
  GenerateNoncesRequest nonces_request;
  GenerateNoncesResponse nonces_response;
  nonces_request.set_nonces_count(3);
  grpc::Status generate_nonces_status =
      stub_->GenerateNonces(&nonces_context, nonces_request, &nonces_response);
  ASSERT_TRUE(generate_nonces_status.ok())
      << "GenerateNonces status code was: "
      << generate_nonces_status.error_code();
  ASSERT_THAT(nonces_response.nonces(), SizeIs(3));

  std::string reencryption_public_key = "";
  std::string ciphertext_associated_data = "ciphertext associated data";

  std::string message_1 = "12";
  absl::StatusOr<Record> rewrapped_record_1 =
      crypto_test_utils::CreateRewrappedRecord(
          message_1, ciphertext_associated_data, recipient_public_key,
          nonces_response.nonces(0), reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_1.ok()) << rewrapped_record_1.status();

  std::string message_2 = "345";
  absl::StatusOr<Record> rewrapped_record_2 =
      crypto_test_utils::CreateRewrappedRecord(
          message_2, ciphertext_associated_data, recipient_public_key,
          nonces_response.nonces(1), reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_2.ok()) << rewrapped_record_2.status();

  std::string message_3 = "6789";
  absl::StatusOr<Record> rewrapped_record_3 =
      crypto_test_utils::CreateRewrappedRecord(
          message_3, ciphertext_associated_data, recipient_public_key,
          nonces_response.nonces(2), reencryption_public_key);
  ASSERT_TRUE(rewrapped_record_3.ok()) << rewrapped_record_3.status();

  TransformRequest transform_request;
  transform_request.add_inputs()->CopyFrom(*rewrapped_record_1);
  transform_request.add_inputs()->CopyFrom(*rewrapped_record_2);
  transform_request.add_inputs()->CopyFrom(*rewrapped_record_3);

  grpc::ClientContext transform_context;
  TransformResponse transform_response;
  grpc::Status transform_status = stub_->Transform(
      &transform_context, transform_request, &transform_response);

  ASSERT_TRUE(transform_status.ok())
      << "Transform status code was: " << transform_status.error_code();
  ASSERT_EQ(transform_response.outputs_size(), 1);
  ASSERT_TRUE(transform_response.outputs(0).has_unencrypted_data());
  ASSERT_EQ(transform_response.outputs(0).unencrypted_data(), "123456789");
}

TEST_F(TestConcatPipelineTransformTest, TransformBeforeConfigureAndAttest) {
  grpc::ClientContext context;
  TransformRequest request;
  TransformResponse response;
  auto status = stub_->Transform(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(status.error_message(),
              HasSubstr("ConfigureAndAttest must be called before Transform"));
}

TEST_F(TestConcatPipelineTransformTest,
       GenerateNoncesBeforeConfigureAndAttest) {
  grpc::ClientContext nonces_context;
  GenerateNoncesRequest nonces_request;
  GenerateNoncesResponse nonces_response;
  nonces_request.set_nonces_count(1);
  auto status =
      stub_->GenerateNonces(&nonces_context, nonces_request, &nonces_response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  ASSERT_THAT(
      status.error_message(),
      HasSubstr("ConfigureAndAttest must be called before GenerateNonces"));
}

}  // namespace

}  // namespace confidential_federated_compute::test_concat
