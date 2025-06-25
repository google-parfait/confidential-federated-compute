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

#include <cstdlib>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/containers/sdk/orchestrator_client.h"
#include "grpcpp/channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "kernel.h"
#include "test_cuda.grpc.pb.h"
#include "test_cuda.pb.h"

namespace confidential_federated_compute::minimum_rules_cuda {
namespace {

using ::grpc::Server;
using ::oak::containers::sdk::OrchestratorClient;
using ::test_cuda::TestRequest;
using ::test_cuda::TestResponse;
using ::test_cuda::TestService;

// Test service to run in the TEE that calls GenerateOutput() and returns the
// result.
class TestServiceImpl : public TestService::Service {
 public:
  TestServiceImpl() {}
  ~TestServiceImpl() override {}

  virtual grpc::Status TestCall(::grpc::ServerContext *context,
                                const TestRequest *request,
                                TestResponse *response) override {
    launch();
    response->set_msg("Cuda kernel call attempted. See log for details.");
    return grpc::Status::OK;
  }
};

void RunServer() {
  std::string server_address("[::]:8080");
  TestServiceImpl service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server = builder.BuildAndStart();
  LOG(INFO) << "Test CUDA server listening on " << server_address << "\n";
  CHECK_OK(OrchestratorClient().NotifyAppReady());
  server->Wait();
}

}  // namespace
}  // namespace confidential_federated_compute::minimum_rules_cuda

int main() {
  confidential_federated_compute::minimum_rules_cuda::RunServer();
  return 0;
}
