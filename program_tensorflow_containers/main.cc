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
#include "cc/containers/sdk/orchestrator_client.h"
#include "cc/containers/sdk/signing_key_handle.h"
#include "confidential_transform_server.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "program_executor_tee/python_manager.h"

namespace confidential_federated_compute::tensorflow::program_executor_tee {

namespace {

using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::containers::sdk::OrchestratorClient;

// Increase gRPC message size limit to 2GB.
static constexpr int kChannelMaxMessageSize = 2 * 1000 * 1000 * 1000;

void RunServer() {
  std::string server_address("[::]:8080");

  TensorflowProgramExecutorTeeConfidentialTransform service(
      std::make_unique<oak::containers::sdk::InstanceSigningKeyHandle>());
  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(kChannelMaxMessageSize);
  builder.SetMaxSendMessageSize(kChannelMaxMessageSize);

  // Be very permissive in allowing client side pings to keep grpc channels
  // open.
  builder.AddChannelArgument(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
  builder.AddChannelArgument(
      GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 0);
  builder.AddChannelArgument(GRPC_ARG_HTTP2_MAX_PING_STRIKES, 0);

  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server = builder.BuildAndStart();
  LOG(INFO) << "Tensorflow Program Executor Confidential Transform Server "
               "listening on "
            << server_address << "\n";

  CHECK_OK(OrchestratorClient().NotifyAppReady());
  server->Wait();
}

}  // namespace

}  // namespace confidential_federated_compute::tensorflow::program_executor_tee

int main(int argc, char** argv) {
  setenv("PYTHONPATH", CPP_PYTHON_PATH, true);
  PythonManager::GetInstance().Start();
  confidential_federated_compute::tensorflow::program_executor_tee::RunServer();
  PythonManager::GetInstance().Stop();
  return 0;
}
