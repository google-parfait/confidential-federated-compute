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
#include "cc/containers/sdk/encryption_key_handle.h"
#include "cc/containers/sdk/orchestrator_client.h"
#include "cc/containers/sdk/signing_key_handle.h"
#include "containers/fed_sql/confidential_transform_server.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::containers::sdk::InstanceEncryptionKeyHandle;
using ::oak::containers::sdk::InstanceSigningKeyHandle;
using ::oak::containers::sdk::OrchestratorClient;

// Increase gRPC message size limit to 2GB.
static constexpr int kChannelMaxMessageSize = 2 * 1000 * 1000 * 1000;

void RunServer() {
  std::string server_address("[::]:8080");

  FedSqlConfidentialTransform service(
      std::make_unique<InstanceSigningKeyHandle>(),
      std::make_unique<InstanceEncryptionKeyHandle>());
  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(kChannelMaxMessageSize);
  builder.SetMaxSendMessageSize(kChannelMaxMessageSize);
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server = builder.BuildAndStart();
  LOG(INFO) << "FedSQL Confidential Transform Server listening on "
            << server_address << "\n";

  CHECK_OK(OrchestratorClient().NotifyAppReady());
  server->Wait();
}

}  // namespace

}  // namespace
   // confidential_federated_compute::fed_sql

int main(int argc, char** argv) {
  confidential_federated_compute::fed_sql::RunServer();
  return 0;
}
