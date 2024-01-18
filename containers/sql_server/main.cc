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

#include "containers/sql_server/pipeline_transform_server.h"
#include "oak_containers/proto/interfaces.grpc.pb.h"
#include "oak_containers/proto/interfaces.pb.h"

namespace confidential_federated_compute::sql_server {

namespace {

using ::google::protobuf::Empty;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::containers::Orchestrator;

void RunServer() {
  std::string server_address("[::]:8080");

  SqlPipelineTransform service;
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server = builder.BuildAndStart();
  std::cout << "SQL Server listening on " << server_address << "\n";

  // The Oak Orchestrator gRPC service is listening on a UDS path. See
  // https://github.com/project-oak/oak/blob/55901b8a4c898c00ecfc14ef4bc65f30cd31d6a9/oak_containers_hello_world_trusted_app/src/orchestrator_client.rs#L45
  grpc::ClientContext context;
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
      "unix:/oak_utils/orchestrator_ipc", grpc::InsecureChannelCredentials());
  std::unique_ptr<Orchestrator::Stub> stub = Orchestrator::NewStub(channel);
  Empty empty_request;
  Empty empty_response;
  stub->NotifyAppReady(&context, empty_request, &empty_response);

  std::cout << "Notified Oak orchestrator app ready" << "\n";

  server->Wait();
}

}  // namespace

}  // namespace confidential_federated_compute::sql_server

int main(int argc, char** argv) {
  confidential_federated_compute::sql_server::RunServer();
  return 0;
}
