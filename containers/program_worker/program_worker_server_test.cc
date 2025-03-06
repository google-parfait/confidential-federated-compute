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
#include "containers/program_worker/program_worker_server.h"

#include <fstream>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "fcp/protos/confidentialcompute/program_worker.grpc.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "google/protobuf/struct.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "proto/containers/orchestrator_crypto_mock.grpc.pb.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::program_worker {

namespace {

using ::fcp::confidentialcompute::ComputationRequest;
using ::fcp::confidentialcompute::ComputationResponse;
using ::fcp::confidentialcompute::ProgramWorker;
using ::fcp::confidentialcompute::TffSessionConfig;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::StatusCode;
using ::oak::containers::v1::MockOrchestratorCryptoStub;
using ::tensorflow_federated::v0::Value;
using ::testing::Test;

constexpr absl::string_view kNoArgumentComputationPath =
    "containers/program_worker/testing/no_argument_comp.txtpb";
constexpr absl::string_view kNoArgumentComputationExpectedResultPath =
    "containers/program_worker/testing/no_argument_comp_expected_result.txtpb";
constexpr absl::string_view kServerDataCompPath =
    "containers/program_worker/testing/server_data_comp.txtpb";
constexpr absl::string_view kServerDataPath =
    "containers/program_worker/testing/server_data.txtpb";
constexpr absl::string_view kServerDataCompExpectedResultPath =
    "containers/program_worker/testing/server_data_comp_expected_result.txtpb";

absl::StatusOr<Value> LoadFileAsTffValue(absl::string_view path,
                                         bool is_computation = true) {
  // Before creating the std::ifstream, convert the absl::string_view to
  // std::string.
  std::string path_str(path);
  std::ifstream file_istream(path_str);
  if (!file_istream) {
    return absl::FailedPreconditionError("Error loading file: " + path_str);
  }
  std::stringstream file_stream;
  file_stream << file_istream.rdbuf();
  if (is_computation) {
    federated_language::Computation computation;
    if (!google::protobuf::TextFormat::ParseFromString(
            std::move(file_stream.str()), &computation)) {
      return absl::InvalidArgumentError(
          "Error parsing TFF Computation from file.");
    }
    Value value;
    *value.mutable_computation() = std::move(computation);
    return value;
  } else {
    Value value;
    if (!google::protobuf::TextFormat::ParseFromString(
            std::move(file_stream.str()), &value)) {
      return absl::InvalidArgumentError(
          "Error parsing TFF Federated from file.");
    }
    return value;
  }
}

class ProgramWorkerTeeServerTest : public Test {
 public:
  ProgramWorkerTeeServerTest() {
    int port;
    const std::string server_address = "[::1]:";
    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    stub_ = ProgramWorker::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  ~ProgramWorkerTeeServerTest() override {
    if (server_ != nullptr) {
      server_->Shutdown();
      server_->Wait();
    }
  }

 protected:
  testing::NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub_;
  ProgramWorkerTee service_{&mock_crypto_stub_};
  std::unique_ptr<Server> server_;
  std::unique_ptr<ProgramWorker::Stub> stub_;
};

TEST_F(ProgramWorkerTeeServerTest, ExecuteReturnsInvalidArgumentError) {
  grpc::ClientContext context;
  ComputationRequest request;
  google::protobuf::Value string_value;
  string_value.set_string_value("test");
  request.mutable_computation()->PackFrom(string_value);
  ComputationResponse response;

  auto status = stub_->Execute(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(ProgramWorkerTeeServerTest, ExecuteNoArgumentComputationReturnsResult) {
  grpc::ClientContext context;
  ComputationRequest request;

  TffSessionConfig comp_request;
  auto function = LoadFileAsTffValue(kNoArgumentComputationPath);
  ASSERT_TRUE(function.ok());
  *comp_request.mutable_function() = *function;
  comp_request.set_num_clients(3);
  comp_request.set_output_access_policy_node_id(1);
  comp_request.set_max_concurrent_computation_calls(1);
  request.mutable_computation()->PackFrom(comp_request);
  ComputationResponse response;

  auto status = stub_->Execute(&context, request, &response);
  ASSERT_TRUE(status.ok());

  Value result;
  ASSERT_TRUE(response.result().UnpackTo(&result));
  auto expected_result =
      LoadFileAsTffValue(kNoArgumentComputationExpectedResultPath, false);
  ASSERT_TRUE(expected_result.ok());
  ASSERT_EQ(result.SerializeAsString(), expected_result->SerializeAsString());
}

TEST_F(ProgramWorkerTeeServerTest, ExecuteServerDataCompReturnsResult) {
  grpc::ClientContext context;
  ComputationRequest request;
  TffSessionConfig comp_request;
  auto function = LoadFileAsTffValue(kServerDataCompPath);
  ASSERT_TRUE(function.ok());
  *comp_request.mutable_function() = *function;
  auto arg = LoadFileAsTffValue(kServerDataPath, false);
  ASSERT_TRUE(arg.ok());
  *comp_request.mutable_initial_arg() = *arg;
  comp_request.set_num_clients(3);
  comp_request.set_output_access_policy_node_id(1);
  comp_request.set_max_concurrent_computation_calls(1);
  request.mutable_computation()->PackFrom(comp_request);
  ComputationResponse response;

  auto status = stub_->Execute(&context, request, &response);
  ASSERT_TRUE(status.ok());

  Value result;
  ASSERT_TRUE(response.result().UnpackTo(&result));
  auto expected_result =
      LoadFileAsTffValue(kServerDataCompExpectedResultPath, false);
  ASSERT_TRUE(expected_result.ok());
  ASSERT_EQ(result.SerializeAsString(), expected_result->SerializeAsString());
}

}  // namespace

}  // namespace confidential_federated_compute::program_worker
