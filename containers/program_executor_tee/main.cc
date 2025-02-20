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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "containers/oak_orchestrator_client.h"
#include "containers/program_executor_tee/confidential_transform_server.h"
#include "grpcpp/channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "proto/containers/interfaces.grpc.pb.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::containers::Orchestrator;
using ::oak::containers::v1::OrchestratorCrypto;

// Increase gRPC message size limit to 2GB.
static constexpr int kChannelMaxMessageSize = 2 * 1000 * 1000 * 1000;

void RunServer() {
  std::string server_address("[::]:8080");
  std::shared_ptr<grpc::Channel> orchestrator_channel =
      CreateOakOrchestratorChannel();

  OrchestratorCrypto::Stub orchestrator_crypto_stub(orchestrator_channel);
  ProgramExecutorTeeConfidentialTransform service(&orchestrator_crypto_stub);
  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(kChannelMaxMessageSize);
  builder.SetMaxSendMessageSize(kChannelMaxMessageSize);
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server = builder.BuildAndStart();
  LOG(INFO) << "Program Executor Confidential Transform Server listening on "
            << server_address << "\n";

  Orchestrator::Stub orchestrator_stub(orchestrator_channel);
  OakOrchestratorClient oak_orchestrator_client(&orchestrator_stub);
  CHECK_OK(oak_orchestrator_client.NotifyAppReady());
  server->Wait();
}

}  // namespace

}  // namespace
   // confidential_federated_compute::program_executor_tee

int main(int argc, char** argv) {
  confidential_federated_compute::program_executor_tee::RunServer();
  return 0;
}
