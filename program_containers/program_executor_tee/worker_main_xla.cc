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
#include "cc/ffi/bytes_bindings.h"
#include "cc/ffi/bytes_view.h"
#include "cc/ffi/error_bindings.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "federated_language_jax/executor/xla_executor.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "program_executor_tee/program_worker_server.h"
#include "proto/session/session.pb.h"

extern "C" {
extern ::oak::session::SessionConfig* create_session_config();
extern void init_tokio_runtime();
}
namespace confidential_federated_compute::program_executor_tee {

namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::containers::sdk::OrchestratorClient;
using ::oak::session::AttestationType;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;

// Increase gRPC message size limit to 2GB
static constexpr int kChannelMaxMessageSize = 2 * 1000 * 1000 * 1000;

absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
CreateExecutor() {
  return federated_language_jax::CreateXLAExecutor();
}

void RunServer() {
  std::string server_address("[::]:8080");

  // Initialize the Rust runtime to create the session config.
  init_tokio_runtime();
  auto* session_config = create_session_config();
  auto service = ProgramWorkerTee::Create(session_config, CreateExecutor);
  CHECK_OK(service) << "Failed to create ProgramWorkerTee service: "
                    << service.status();

  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(kChannelMaxMessageSize);
  builder.SetMaxSendMessageSize(kChannelMaxMessageSize);
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service->get());
  std::unique_ptr<Server> server = builder.BuildAndStart();
  LOG(INFO) << "Program Worker Server listening on " << server_address << "\n";

  CHECK_OK(OrchestratorClient().NotifyAppReady());
  server->Wait();
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee

int main(int argc, char** argv) {
  confidential_federated_compute::program_executor_tee::RunServer();
  return 0;
}