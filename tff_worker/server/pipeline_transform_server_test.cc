/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tff_worker/server/pipeline_transform_server.h"

#include <fstream>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "fcp/protos/confidentialcompute/tff_worker_configuration.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/status.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::tff_worker {

namespace {

using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::TffWorkerConfiguration;
using ::fcp::confidentialcompute::TffWorkerConfiguration_ClientWork;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::StatusCode;
using ::testing::HasSubstr;
using ::testing::Test;


constexpr absl::string_view kClientWorkComputationPath =
    "tff_worker/server/testing/client_work_computation.pb";


absl::StatusOr<std::string> LoadFileAsString(absl::string_view path) {
  std::string path_str(path);  // Convert absl::string_view to std::string.
  std::ifstream file_istream(path_str);
  if (!file_istream) {
    return absl::FailedPreconditionError(
        "Error loading file: " + std::string(path));
  }
  std::stringstream file_stream;
  file_stream << file_istream.rdbuf();
  return file_stream.str();
}


class TffPipelineTransformTest : public Test {
 public:
  void SetUp() override {
    int port;
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address +
              std::to_string(port);
    channel_ = grpc::CreateChannel(
        server_address + std::to_string(port),
        grpc::experimental::LocalCredentials(LOCAL_TCP));
    stub_ = PipelineTransform::NewStub(channel_);
  }

  void TearDown() override { server_->Shutdown(); }

 protected:
  TffPipelineTransform service_;
  std::unique_ptr<Server> server_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<PipelineTransform::Stub> stub_;
};

TEST_F(TffPipelineTransformTest, InvalidConfigureAndAttestRequest) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              HasSubstr("must contain configuration"));
}

TEST_F(TffPipelineTransformTest,
       MissingTffWorkerConfigurationConfigureAndAttestRequest) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;
  TffWorkerConfiguration_ClientWork wrong_proto;
  request.mutable_configuration()->PackFrom(wrong_proto);
  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(
      status.error_message(),
      HasSubstr(
          "must be a tff_worker_configuration_pb2.TffWorkerConfiguration"));
}

TEST_F(TffPipelineTransformTest, ConfigureAndAttestMoreThanOnce) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;

  // Populate test TffWorkerConfiguration with a computation that has client
  // work.
  TffWorkerConfiguration tff_worker_configuration;
  TffWorkerConfiguration_ClientWork* client_work =
      tff_worker_configuration.mutable_client_work();
  absl::StatusOr<std::string> serialized_proto_data = LoadFileAsString(
      kClientWorkComputationPath);
  ASSERT_TRUE(serialized_proto_data.ok()) << serialized_proto_data.status();
  client_work->set_serialized_client_work_computation(*serialized_proto_data);
  request.mutable_configuration()->PackFrom(tff_worker_configuration);

  auto first_status = stub_->ConfigureAndAttest(&context, request, &response);
  ASSERT_TRUE(first_status.ok()) << first_status.error_message();

  grpc::ClientContext second_context;
  auto second_status = stub_->ConfigureAndAttest(
      &second_context, request, &response);
  EXPECT_EQ(second_status.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_THAT(second_status.error_message(),
              HasSubstr("ConfigureAndAttest can only be called once"));
}

TEST_F(TffPipelineTransformTest, ValidConfigureAndAttest) {
  grpc::ClientContext context;
  ConfigureAndAttestRequest request;
  ConfigureAndAttestResponse response;

  // Populate test TffWorkerConfiguration with a computation that has client
  // work.
  TffWorkerConfiguration tff_worker_configuration;
  TffWorkerConfiguration_ClientWork* client_work =
      tff_worker_configuration.mutable_client_work();
  absl::StatusOr<std::string> serialized_proto_data = LoadFileAsString(
      kClientWorkComputationPath);
  ASSERT_TRUE(serialized_proto_data.ok()) << serialized_proto_data.status();
  client_work->set_serialized_client_work_computation(*serialized_proto_data);
  request.mutable_configuration()->PackFrom(tff_worker_configuration);

  auto status = stub_->ConfigureAndAttest(&context, request, &response);
  EXPECT_TRUE(status.ok()) << status.error_message();
}

}  // namespace

}  // namespace confidential_federated_compute::tff_worker
