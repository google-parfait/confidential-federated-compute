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

#include "containers/program_executor_tee/program_context/cc/computation_runner.h"

#include <optional>

#include "absl/status/statusor.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/tff_execution_helper.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

// Creates a non-distributed TFF execution stack.
absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>> CreateExecutor(
    int num_clients) {
  auto leaf_executor_fn =
      []() -> absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>> {
    return tensorflow_federated::CreateReferenceResolvingExecutor(
        tensorflow_federated::CreateTensorFlowExecutor());
  };
  tensorflow_federated::CardinalityMap cardinality_map;
  cardinality_map[tensorflow_federated::kClientsUri] = num_clients;
  FCP_ASSIGN_OR_RETURN(auto server_child, leaf_executor_fn());
  FCP_ASSIGN_OR_RETURN(auto client_child, leaf_executor_fn());
  FCP_ASSIGN_OR_RETURN(auto federating_executor,
                       tensorflow_federated::CreateFederatingExecutor(
                           server_child, client_child, cardinality_map));
  return tensorflow_federated::CreateReferenceResolvingExecutor(
      federating_executor);
}

// Creates a distributed TFF execution stack.
absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
CreateDistributedExecutor(const std::vector<std::string>& worker_bns,
                          int num_clients) {
  return absl::UnimplementedError(
      "Distributed execution is not supported yet.");
}

// Executes comp(arg) on the provided TFF execution stack.
absl::StatusOr<tensorflow_federated::v0::Value> Execute(
    std::shared_ptr<tensorflow_federated::Executor> executor,
    tensorflow_federated::v0::Value comp,
    std::optional<tensorflow_federated::v0::Value> arg) {
  FCP_ASSIGN_OR_RETURN(tensorflow_federated::OwnedValueId fn_handle,
                       executor->CreateValue(comp));
  tensorflow_federated::v0::Value call_result;
  if (arg.has_value()) {
    FCP_ASSIGN_OR_RETURN(
        std::shared_ptr<tensorflow_federated::OwnedValueId> arg_handle,
        fcp::confidential_compute::Embed(*arg, executor));
    FCP_ASSIGN_OR_RETURN(tensorflow_federated::OwnedValueId call_handle,
                         executor->CreateCall(fn_handle, *arg_handle));
    FCP_RETURN_IF_ERROR(executor->Materialize(call_handle, &call_result));
  } else {
    FCP_ASSIGN_OR_RETURN(tensorflow_federated::OwnedValueId call_handle,
                         executor->CreateCall(fn_handle, std::nullopt));
    FCP_RETURN_IF_ERROR(executor->Materialize(call_handle, &call_result));
  }
  return call_result;
}

}  // namespace

absl::StatusOr<tensorflow_federated::v0::Value> ComputationRunner::InvokeComp(
    int num_clients, const tensorflow_federated::v0::Value comp,
    std::optional<tensorflow_federated::v0::Value> arg) {
  FCP_ASSIGN_OR_RETURN(
      auto executor, worker_bns_.empty()
                         ? CreateExecutor(num_clients)
                         : CreateDistributedExecutor(worker_bns_, num_clients));
  return Execute(executor, std::move(comp), std::move(arg));
}

}  // namespace confidential_federated_compute::program_executor_tee
