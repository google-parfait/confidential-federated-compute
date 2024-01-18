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

#include <string>

#include "absl/log/log.h"
#include "containers/oak_orchestrator_client.h"
#include "containers/test_concat/pipeline_transform_server.h"
#include "oak_containers/proto/interfaces.grpc.pb.h"
#include "oak_containers/proto/interfaces.pb.h"

namespace confidential_federated_compute::test_concat {

namespace {

using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::containers::Orchestrator;

void RunServer() {
  std::string server_address("[::]:8080");

  TestConcatPipelineTransform service;
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server = builder.BuildAndStart();
  LOG(INFO) << "Test Concat Server listening on " << server_address << "\n";

  std::unique_ptr<Orchestrator::Stub> orchestrator_stub =
      CreateOakOrchestratorStub();
  OakOrchestratorClient oak_orchestrator_client(orchestrator_stub.get());
  absl::Status oak_notify_status = oak_orchestrator_client.NotifyAppReady();
  FCP_CHECK(oak_notify_status.ok()) << oak_notify_status;
  server->Wait();
}

}  // namespace

}  // namespace confidential_federated_compute::test_concat

int main(int argc, char** argv) {
  confidential_federated_compute::test_concat::RunServer();
  return 0;
}
