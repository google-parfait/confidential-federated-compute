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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_OAK_ORCHESTRATOR_CLIENT_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_OAK_ORCHESTRATOR_CLIENT_H_

#include <memory>

#include "absl/status/status.h"
#include "grpcpp/channel.h"
#include "proto/containers/interfaces.grpc.pb.h"

namespace confidential_federated_compute {

// Creates a channel for interacting with the Oak Orchestrator.
std::shared_ptr<grpc::Channel> CreateOakOrchestratorChannel();

// Client for interacting with the Oak Containers Orchestrator service.
// See https://github.com/project-oak/oak/tree/main/oak_containers_orchestrator
class OakOrchestratorClient {
 public:
  // Constructs the OakOrchestratorClient from a pointer to an Oak Orchestrator
  // stub.
  // The stub must outlive this class.
  explicit OakOrchestratorClient(oak::containers::Orchestrator::StubInterface*);

  // Notifies the Oak Orchestrator that this container application has been
  // launched successfully and is ready to serve requests.
  absl::Status NotifyAppReady();

 private:
  oak::containers::Orchestrator::StubInterface* stub_;
};

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_OAK_ORCHESTRATOR_CLIENT_H_
