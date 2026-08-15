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

// Stateful helper class for executing TFF computations over gRPC.
//
// Supports three primary execution modes:
// 1. Distributing unpartitioned federated computations to multiple workers
//    via a C++ composing executor stack (e.g., ComposingExecutor or
//    ElasticComposingExecutor) when worker_bns is configured and
//    use_mergeable_execution_context is false.
// 2. Executing work on a single remote worker after work distribution has
//    already been handled upstream by MergeableCompExecutionContext (when
//    use_mergeable_execution_context is true and incoming requests specify a
//    single target worker_bns).
// 3. Executing work on a local server stack when no remote workers are
//    configured (worker_bns is empty) or when the request's worker_bns is
//    empty (e.g., merge and after_merge computations under
//    MergeableCompExecutionContext).
class ComputationRunner : public fcp::confidentialcompute::outgoing::
                              ComputationDelegation::Service {
 public:
  ComputationRunner(
      std::function<
          absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
          leaf_executor_factory,
      std::vector<std::string> worker_bns,
      std::string serialized_reference_values,
      std::string outgoing_server_address, bool use_elastic_composing_executor,
      bool use_mergeable_execution_context);

  // Executes the TFF computation represented in the request message using a C++
  // execution stack. Returns a tensorflow_federated::v0::Value in the response
  // message.
  //
  // Work is dispatched according to the configuration and request:
  // - If request->worker_bns() is specified, delegates execution to that
  //   specific worker via a remote executor (used when worker distribution was
  //   already performed by MergeableCompExecutionContext).
  // - If request->worker_bns() is empty:
  //   - When worker_bns_ is empty or use_mergeable_execution_context is true,
  //     executes on the local server stack.
  //   - Otherwise, distributes the computation across all workers in
  //     worker_bns_ using ComposingExecutor or ElasticComposingExecutor.
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
  // Whether to use the ElasticComposingExecutor or ComposingExecutor.
  bool use_elastic_composing_executor_;
  // Whether to use the MergeableCompExecutionContext.
  bool use_mergeable_execution_context_;
  // ComputationDelegation service stub for communication with workers. This is
  // only initialized if worker_bns_ is non-empty.
  std::unique_ptr<
      fcp::confidentialcompute::outgoing::ComputationDelegation::Stub>
      stub_;
  std::vector<std::shared_ptr<NoiseClientSession>> noise_client_sessions_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_COMPUTATION_RUNNER_H_
