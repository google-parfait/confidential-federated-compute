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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_WORKER_PROGRAM_WORKER_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_WORKER_PROGRAM_WORKER_SERVER_H_

#include "absl/log/die_if_null.h"
#include "fcp/protos/confidentialcompute/program_worker.grpc.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"

namespace confidential_federated_compute::program_worker {

// ProgramWorker service implementation.
class ProgramWorkerTee
    : public fcp::confidentialcompute::ProgramWorker::Service {
 public:
  ProgramWorkerTee(
      oak::containers::v1::OrchestratorCrypto::StubInterface* crypto_stub)
      : crypto_stub_(*ABSL_DIE_IF_NULL(crypto_stub)) {}

  grpc::Status Execute(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::ComputationRequest* request,
      fcp::confidentialcompute::ComputationResponse* response);

 private:
  oak::containers::v1::OrchestratorCrypto::StubInterface& crypto_stub_;
};

}  // namespace confidential_federated_compute::program_worker

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_WORKER_PROGRAM_WORKER_SERVER_H_
