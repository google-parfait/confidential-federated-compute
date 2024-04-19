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

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "containers/oak_orchestrator_client.h"
#include "containers/test_concat/pipeline_transform_server.h"
#include "grpcpp/channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "proto/containers/interfaces.grpc.pb.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"

namespace confidential_federated_compute::test_concat {

namespace {

using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::containers::Orchestrator;
using ::oak::containers::v1::OrchestratorCrypto;

void RunServer() {
  std::string server_address("[::]:8080");
  std::shared_ptr<grpc::Channel> orchestrator_channel =
      CreateOakOrchestratorChannel();

  OrchestratorCrypto::Stub orchestrator_crypto_stub(orchestrator_channel);
  TestConcatPipelineTransform service(&orchestrator_crypto_stub);
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server = builder.BuildAndStart();
  LOG(INFO) << "Test Concat Server listening on " << server_address << "\n";

  Orchestrator::Stub orchestrator_stub(orchestrator_channel);
  OakOrchestratorClient oak_orchestrator_client(&orchestrator_stub);
  CHECK_OK(oak_orchestrator_client.NotifyAppReady());
  server->Wait();
}

}  // namespace

}  // namespace confidential_federated_compute::test_concat

int main(int argc, char** argv) {
  confidential_federated_compute::test_concat::RunServer();
  return 0;
}
