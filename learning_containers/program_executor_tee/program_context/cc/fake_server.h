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

#include <grpcpp/grpcpp.h>

#include "grpcpp/server_context.h"
#include "program_executor_tee/program_context/cc/fake_computation_delegation_service.h"
#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"

namespace confidential_federated_compute::program_executor_tee {

namespace py = pybind11;

class FakeServer {
 public:
  FakeServer(int port, FakeDataReadWriteService* data_read_write_service,
             FakeComputationDelegationService* computation_delegation_service)
      : server_address_("[::1]:" + std::to_string(port)),
        data_read_write_service_(data_read_write_service),
        computation_delegation_service_(computation_delegation_service) {}

  void Start() {
    if (server_) {
      throw std::runtime_error("Server is already running.");
    }
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_,
                             grpc::InsecureServerCredentials());
    if (data_read_write_service_) {
      builder.RegisterService(data_read_write_service_);
    }
    if (computation_delegation_service_) {
      builder.RegisterService(computation_delegation_service_);
    }
    server_ = builder.BuildAndStart();
    if (!server_) {
      throw std::runtime_error("Could not start server on " + server_address_);
    }
  }

  void Stop() {
    if (!server_) {
      return;  // It's often better to make Stop idempotent.
    }
    server_->Shutdown();
  }

  // The destructor will ensure Stop() is called.
  ~FakeServer() {
    if (server_) {
      Stop();
    }
  }

 private:
  std::string server_address_;
  FakeDataReadWriteService* data_read_write_service_;
  FakeComputationDelegationService* computation_delegation_service_;
  std::unique_ptr<grpc::Server> server_;
};

}  // namespace confidential_federated_compute::program_executor_tee