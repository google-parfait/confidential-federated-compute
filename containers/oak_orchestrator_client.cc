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
#include "containers/oak_orchestrator_client.h"

#include <iostream>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/empty.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/support/status.h"
#include "oak_containers/proto/interfaces.grpc.pb.h"
#include "oak_containers/proto/interfaces.pb.h"

namespace confidential_federated_compute {

using ::google::protobuf::Empty;
using ::oak::containers::Orchestrator;

std::shared_ptr<grpc::Channel> CreateOakOrchestratorChannel() {
  // The Oak Orchestrator gRPC service is listening on a UDS path. See
  // https://github.com/project-oak/oak/blob/55901b8a4c898c00ecfc14ef4bc65f30cd31d6a9/oak_containers_hello_world_trusted_app/src/orchestrator_client.rs#L45
  return grpc::CreateChannel(
      "unix:/oak_utils/orchestrator_ipc", grpc::InsecureChannelCredentials());
}

OakOrchestratorClient::OakOrchestratorClient(Orchestrator::StubInterface* stub)
    : stub_(stub) {}

absl::Status OakOrchestratorClient::NotifyAppReady() {
  grpc::ClientContext context;
  Empty empty_request;
  Empty empty_response;
  grpc::Status status =
      stub_->NotifyAppReady(&context, empty_request, &empty_response);
  if (status.ok()) {
    LOG(INFO) << "Notified Oak Orchestrator that application is ready\n";
    return absl::OkStatus();
  } else {
    return absl::InternalError(
        absl::StrCat("Failed to notify Oak Orchestrator of app readiness. "
                     "Orchestrator returned error status ",
                     status.error_code(), ": ", status.error_message()));
  }
}

}  // namespace confidential_federated_compute
