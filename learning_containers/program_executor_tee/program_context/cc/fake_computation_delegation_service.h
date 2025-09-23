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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_FAKE_COMPUTATION_DELEGATION_SERVICE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_FAKE_COMPUTATION_DELEGATION_SERVICE_H_

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "cc/oak_session/server_session.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_context.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace confidential_federated_compute::program_executor_tee {

class FakeComputationDelegationService
    : public fcp::confidentialcompute::outgoing::ComputationDelegation::
          Service {
 public:
  FakeComputationDelegationService(
      std::vector<std::string> worker_bns,
      std::function<
          absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
          leaf_executor_factory);

  grpc::Status Execute(
      ::grpc::ServerContext* context,
      const ::fcp::confidentialcompute::outgoing::ComputationRequest* request,
      ::fcp::confidentialcompute::outgoing::ComputationResponse* response)
      override;

 private:
  absl::StatusOr<oak::session::v1::PlaintextMessage> DecryptRequest(
      const oak::session::v1::SessionRequest& session_request,
      std::string worker_bns);

  absl::StatusOr<oak::session::v1::SessionResponse> EncryptResult(
      const oak::session::v1::PlaintextMessage& plaintext_response,
      std::string worker_bns);

  absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
  InitializeTffExecutor(
      const ::fcp::confidentialcompute::TffSessionConfig& comp_request);

  // Function for generating a leaf executor.
  std::function<
      absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
      leaf_executor_factory_;
  // A map from worker bns to the noise server session for the worker.
  absl::flat_hash_map<std::string, std::unique_ptr<oak::session::ServerSession>>
      server_sessions_;
  absl::Mutex mutex_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_FAKE_COMPUTATION_DELEGATION_SERVICE_H_