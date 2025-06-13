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

#include "absl/synchronization/mutex.h"
#include "containers/program_executor_tee/program_context/cc/noise_client_session.h"
#include "fcp/confidentialcompute/lambda_runner.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"

namespace confidential_federated_compute::program_executor_tee {

class ComputationDelegationLambdaRunner
    : public fcp::confidential_compute::LambdaRunner {
 public:
  static absl::StatusOr<
      std::shared_ptr<fcp::confidential_compute::LambdaRunner>>
  Create(const std::string& worker_bns,
         oak::session::SessionConfig* session_config,
         std::function<ComputationDelegationResult(
             ::fcp::confidentialcompute::outgoing::ComputationRequest)>
             computation_delegation_proxy) {
    FCP_ASSIGN_OR_RETURN(auto noise_client_session,
                         NoiseClientSession::Create(
                             worker_bns, session_config,
                             std::function(computation_delegation_proxy)));
    if (worker_bns.empty()) {
      return absl::InvalidArgumentError("Worker bns is empty.");
    }
    return std::make_shared<ComputationDelegationLambdaRunner>(
        std::move(noise_client_session), computation_delegation_proxy);
  }

  // Constructor public for testing. Prefer using Create() instead.
  ComputationDelegationLambdaRunner(
      std::shared_ptr<NoiseClientSessionInterface> noise_client_session,
      std::function<ComputationDelegationResult(
          ::fcp::confidentialcompute::outgoing::ComputationRequest)>
          computation_delegation_proxy)
      : noise_client_session_(std::move(noise_client_session)),
        computation_delegation_proxy_(computation_delegation_proxy) {}

  absl::StatusOr<tensorflow_federated::v0::Value> ExecuteComp(
      tensorflow_federated::v0::Value function,
      std::optional<tensorflow_federated::v0::Value> arg,
      int32_t num_clients) override;

 private:
  std::shared_ptr<NoiseClientSessionInterface> noise_client_session_;
  std::function<ComputationDelegationResult(
      ::fcp::confidentialcompute::outgoing::ComputationRequest)>
      computation_delegation_proxy_;
  absl::Mutex mutex_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_DELEGATION_LAMBDA_RUNNER_H_
