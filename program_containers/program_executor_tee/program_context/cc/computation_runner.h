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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_RUNNER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_RUNNER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "program_executor_tee/program_context/cc/noise_client_session.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

// Increase gRPC message size limit to 2GB
inline constexpr int kMaxGrpcMessageSize = 2 * 1000 * 1000 * 1000;

// Stateful helper class for executing TFF computations.
class ComputationRunner : public fcp::confidentialcompute::outgoing::
                              ComputationDelegation::Service {
 public:
  ComputationRunner(
      std::function<
          absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
          leaf_executor_factory,
      std::vector<std::string> worker_bns,
      std::string serialized_reference_values,
      std::string outgoing_server_address);

  // Executes the TFF computation represented in the request message using a C++
  // execution stack. Returns a tensorflow_federated::v0::Value in the response
  // message. If worker_bns_ is empty, the computation will be executed in a
  // non-distributed manner, else parts of the computation will be distributed
  // to the workers. Only TF is supported for now.
  // TODO: Add support for XLA execution.
  grpc::Status Execute(
      ::grpc::ServerContext* context,
      const ::fcp::confidentialcompute::outgoing::ComputationRequest* request,
      ::fcp::confidentialcompute::outgoing::ComputationResponse* response)
      override;

 private:
  absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
  CreateDistributedExecutor(
      std::function<
          absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
          leaf_executor_factory,
      int num_clients);

  std::function<
      absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
      leaf_executor_factory_;
  // Addresses of worker machines running the program_worker binary that can be
  // used to execute computations in a distributed manner.
  std::vector<std::string> worker_bns_;
  // ComputationDelegation service stub for communication with workers. This is
  // only initialized if worker_bns_ is non-empty.
  std::unique_ptr<
      fcp::confidentialcompute::outgoing::ComputationDelegation::Stub>
      stub_;
  std::vector<std::shared_ptr<NoiseClientSession>> noise_client_sessions_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_RUNNER_H_
