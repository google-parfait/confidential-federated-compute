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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_NOISE_CLIENT_SESSION_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_NOISE_CLIENT_SESSION_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cc/oak_session/client_session.h"
#include "cc/oak_session/config.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "proto/session/session.pb.h"

namespace confidential_federated_compute::program_executor_tee {

// Struct to hold both the ComputationResponse and a status.
// This is used to return the result with a status of the computation delegation
// for the gRPC proxy.
struct ComputationDelegationResult {
  ::fcp::confidentialcompute::outgoing::ComputationResponse response;
  grpc::Status status;
};

// Interface for a noise client session that delegates computation requests to a
// worker.
class NoiseClientSessionInterface {
 public:
  virtual ~NoiseClientSessionInterface() = default;
  virtual absl::StatusOr<oak::session::v1::PlaintextMessage>
  DelegateComputation(
      const oak::session::v1::PlaintextMessage& plaintext_request) = 0;
};

// Implementation of NoiseClientSessionInterface for program executor tee.
class NoiseClientSession : public NoiseClientSessionInterface {
 public:
  // Creates a NoiseClientSession.
  static absl::StatusOr<std::shared_ptr<NoiseClientSession>> Create(
      const std::string& worker_bns,
      oak::session::SessionConfig* session_config,
      std::function<ComputationDelegationResult(
          ::fcp::confidentialcompute::outgoing::ComputationRequest)>
          computation_delegation_proxy) {
    FCP_ASSIGN_OR_RETURN(auto client_session,
                         oak::session::ClientSession::Create(session_config));
    return std::shared_ptr<NoiseClientSession>(new NoiseClientSession(
        worker_bns, std::move(client_session), computation_delegation_proxy));
  }

  // Delegates the computation request (serialized as a PlaintextMessage) to the
  // worker, and returns the decrypted response.
  absl::StatusOr<oak::session::v1::PlaintextMessage> DelegateComputation(
      const oak::session::v1::PlaintextMessage& plaintext_request) override;

  // Opens the session before sending any actual computation request.
  absl::Status OpenSession();

 private:
  NoiseClientSession(
      const std::string& worker_bns,
      std::unique_ptr<oak::session::ClientSession> client_session,
      std::function<ComputationDelegationResult(
          ::fcp::confidentialcompute::outgoing::ComputationRequest)>
          computation_delegation_proxy)
      : worker_bns_(worker_bns),
        client_session_(std::move(client_session)),
        computation_delegation_proxy_(computation_delegation_proxy) {}

  absl::StatusOr<oak::session::v1::SessionRequest> EncryptRequest(
      const oak::session::v1::PlaintextMessage& plaintext_request);

  absl::StatusOr<oak::session::v1::PlaintextMessage> DecryptResponse(
      const oak::session::v1::SessionResponse& session_response);

  const std::string worker_bns_;
  std::unique_ptr<oak::session::ClientSession> client_session_;
  std::function<ComputationDelegationResult(
      ::fcp::confidentialcompute::outgoing::ComputationRequest)>
      computation_delegation_proxy_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_NOISE_CLIENT_SESSION_H_
