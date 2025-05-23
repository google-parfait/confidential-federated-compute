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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_DELEGATION_LAMBDA_RUNNER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_DELEGATION_LAMBDA_RUNNER_H_

#include "containers/program_executor_tee/program_context/cc/noise_client_session.h"
#include "fcp/confidentialcompute/lambda_runner.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"

namespace confidential_federated_compute::program_executor_tee {

class ComputationDelegationLambdaRunner
    : public fcp::confidential_compute::LambdaRunner {
 public:
  // Preferred way to create a ComputationDelegationLambdaRunner.
  static absl::StatusOr<
      std::unique_ptr<fcp::confidential_compute::LambdaRunner>>
  Create(
      const std::string& worker_bns,
      oak::session::SessionConfig* session_config,
      fcp::confidentialcompute::outgoing::ComputationDelegation::StubInterface*
          stub) {
    if (worker_bns.empty()) {
      return absl::InvalidArgumentError("Worker bns is empty.");
    }
    FCP_ASSIGN_OR_RETURN(
        auto noise_client_session,
        NoiseClientSession::Create(worker_bns, session_config, stub));
    return absl::WrapUnique(
        new ComputationDelegationLambdaRunner(std::move(noise_client_session)));
  }

  // Constructor public for testing. Prefer using Create() instead.
  ComputationDelegationLambdaRunner(
      std::unique_ptr<NoiseClientSessionInterface> noise_client_session)
      : noise_client_session_(std::move(noise_client_session)) {}

  absl::StatusOr<tensorflow_federated::v0::Value> ExecuteComp(
      tensorflow_federated::v0::Value function,
      std::optional<tensorflow_federated::v0::Value> arg,
      int32_t num_clients) override;

 private:
  std::unique_ptr<NoiseClientSessionInterface> noise_client_session_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_DELEGATION_LAMBDA_RUNNER_H_
