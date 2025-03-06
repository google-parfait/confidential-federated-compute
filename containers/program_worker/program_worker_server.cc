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
#include "containers/program_worker/program_worker_server.h"

#include "absl/status/statusor.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/tff_execution_helper.h"
#include "fcp/protos/confidentialcompute/program_worker.grpc.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_worker {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ComputationRequest;
using ::fcp::confidentialcompute::ComputationResponse;
using ::fcp::confidentialcompute::TffSessionConfig;
using ::grpc::ServerContext;
using ::grpc::Status;

namespace {

absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
InitializeTffExecutor(const TffSessionConfig& comp_request) {
  int32_t max_concurrent_computation_calls =
      comp_request.max_concurrent_computation_calls();
  auto leaf_executor_fn = [max_concurrent_computation_calls]() {
    return tensorflow_federated::CreateReferenceResolvingExecutor(
        tensorflow_federated::CreateTensorFlowExecutor(
            max_concurrent_computation_calls));
  };
  tensorflow_federated::CardinalityMap cardinality_map;
  cardinality_map[tensorflow_federated::kClientsUri] =
      comp_request.num_clients();
  FCP_ASSIGN_OR_RETURN(
      auto federating_executor,
      tensorflow_federated::CreateFederatingExecutor(
          /*server_child=*/leaf_executor_fn(),
          /*client_child=*/leaf_executor_fn(), cardinality_map));
  return tensorflow_federated::CreateReferenceResolvingExecutor(
      federating_executor);
}

}  // namespace

grpc::Status ProgramWorkerTee::Execute(ServerContext* context,
                                       const ComputationRequest* request,
                                       ComputationResponse* response) {
  // We now use the TffSessionConfig as the request message.
  TffSessionConfig comp_request;
  if (!request->computation().UnpackTo(&comp_request)) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "ComputationRequest invalid.");
  }

  // Initialize the TFF executor according to the TffSessionConfig embedded in
  // the request.
  auto child_executor_or = InitializeTffExecutor(comp_request);
  if (!child_executor_or.ok()) {
    return ToGrpcStatus(child_executor_or.status());
  }
  std::shared_ptr<tensorflow_federated::Executor> child_executor =
      std::move(child_executor_or.value());

  tensorflow_federated::v0::Value function = std::move(comp_request.function());
  std::optional<tensorflow_federated::v0::Value> argument = std::nullopt;
  if (comp_request.has_initial_arg()) {
    argument = std::move(comp_request.initial_arg());
  }
  // TODO: b/378243349 - Add support for data pointers input.
  // We now assume that the function and argument are all tff values.
  auto fn_handle_or = child_executor->CreateValue(function);
  if (!fn_handle_or.ok()) {
    return ToGrpcStatus(fn_handle_or.status());
  }

  std::optional<tensorflow_federated::OwnedValueId> optional_arg_handle;
  if (argument.has_value()) {
    auto arg_handle =
        fcp::confidential_compute::Embed(argument.value(), child_executor);
    if (!arg_handle.ok()) {
      return ToGrpcStatus(arg_handle.status());
    }
    optional_arg_handle = std::move(*arg_handle.value());
  }

  auto call_handle_or =
      child_executor->CreateCall(fn_handle_or.value(), optional_arg_handle);
  if (!call_handle_or.ok()) {
    return ToGrpcStatus(call_handle_or.status());
  }
  tensorflow_federated::OwnedValueId call_handle =
      std::move(call_handle_or.value());
  tensorflow_federated::v0::Value call_result;
  auto materialize_status =
      child_executor->Materialize(call_handle, &call_result);
  if (!materialize_status.ok()) {
    return ToGrpcStatus(materialize_status);
  }
  // TODO b/378243349 - Encrypt the result into a blob with the
  // output_access_policy_node_id set in the request.
  response->mutable_result()->PackFrom(call_result);

  return grpc::Status::OK;
}

}  // namespace confidential_federated_compute::program_worker
