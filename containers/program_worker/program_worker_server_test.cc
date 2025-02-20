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

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "fcp/protos/confidentialcompute/program_worker.grpc.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "proto/containers/orchestrator_crypto_mock.grpc.pb.h"

namespace confidential_federated_compute::program_worker {

namespace {

using ::fcp::confidentialcompute::ComputationRequest;
using ::fcp::confidentialcompute::ComputationResponse;
using ::fcp::confidentialcompute::ProgramWorker;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::StatusCode;
using ::oak::containers::v1::MockOrchestratorCryptoStub;
using ::testing::Test;

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

  ~ProgramWorkerTeeServerTest() override { server_->Shutdown(); }

 protected:
  testing::NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub_;
  ProgramWorkerTee service_{&mock_crypto_stub_};
  std::unique_ptr<Server> server_;
  std::unique_ptr<ProgramWorker::Stub> stub_;
};

TEST_F(ProgramWorkerTeeServerTest, ExecuteReturnsUnimplemented) {
  grpc::ClientContext context;
  ComputationRequest request;
  ComputationResponse response;

  auto status = stub_->Execute(&context, request, &response);
  ASSERT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);
}

}  // namespace

}  // namespace confidential_federated_compute::program_worker
