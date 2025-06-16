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

#include <memory>
#include <utility>

#include "absl/log/die_if_null.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/server_session.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/program_worker.grpc.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute::program_worker {

// ProgramWorker service implementation.
class ProgramWorkerTee
    : public fcp::confidentialcompute::ProgramWorker::Service {
 public:
  static absl::StatusOr<std::unique_ptr<ProgramWorkerTee>> Create(
      oak::session::SessionConfig* session_config) {
    FCP_ASSIGN_OR_RETURN(auto server_session,
                         oak::session::ServerSession::Create(session_config));
    return absl::WrapUnique(new ProgramWorkerTee(std::move(server_session)));
  }

  grpc::Status Execute(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::ComputationRequest* request,
      fcp::confidentialcompute::ComputationResponse* response);

 private:
  explicit ProgramWorkerTee(
      std::unique_ptr<oak::session::ServerSession> server_session)
      : server_session_(std::move(server_session)) {}

  absl::StatusOr<oak::session::v1::PlaintextMessage> DecryptRequest(
      const oak::session::v1::SessionRequest& session_request);

  absl::StatusOr<oak::session::v1::SessionResponse> EncryptResult(
      const oak::session::v1::PlaintextMessage& plaintext_response);

  std::unique_ptr<oak::session::ServerSession> server_session_;
};

}  // namespace confidential_federated_compute::program_worker

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_WORKER_PROGRAM_WORKER_SERVER_H_
