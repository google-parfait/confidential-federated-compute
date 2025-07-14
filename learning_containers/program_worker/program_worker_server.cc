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
#include "program_worker/program_worker_server.h"

#include "absl/status/statusor.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/tff_execution_helper.h"
#include "fcp/protos/confidentialcompute/program_worker.grpc.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "proto/session/session.pb.h"
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
using ::oak::session::v1::PlaintextMessage;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

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

absl::StatusOr<PlaintextMessage> ProgramWorkerTee::DecryptRequest(
    const SessionRequest& session_request) {
  FCP_RETURN_IF_ERROR(server_session_->PutIncomingMessage(session_request));
  FCP_ASSIGN_OR_RETURN(auto plaintext_request, server_session_->Read());
  if (!plaintext_request.has_value()) {
    return absl::InvalidArgumentError(
        "Could not read plaintext message from the request.");
  }
  return plaintext_request.value();
}

absl::StatusOr<SessionResponse> ProgramWorkerTee::EncryptResult(
    const PlaintextMessage& plaintext_response) {
  FCP_RETURN_IF_ERROR(server_session_->Write(plaintext_response));
  FCP_ASSIGN_OR_RETURN(auto session_response,
                       server_session_->GetOutgoingMessage());
  if (!session_response.has_value()) {
    return absl::InvalidArgumentError(
        "Could not generate SessionResponse for the request.");
  }
  return session_response.value();
}

grpc::Status ProgramWorkerTee::Execute(ServerContext* context,
                                       const ComputationRequest* request,
                                       ComputationResponse* response) {
  SessionRequest session_request;
  if (!request->computation().UnpackTo(&session_request)) {
    return grpc::Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        "ComputationRequest cannot be unpacked to noise SessionRequest.");
  }

  // Perform handshake until the channel is open.
  if (!server_session_->IsOpen()) {
    auto put_incoming_message_status =
        server_session_->PutIncomingMessage(session_request);
    if (!put_incoming_message_status.ok()) {
      return ToGrpcStatus(put_incoming_message_status);
    }
    absl::StatusOr<std::optional<SessionResponse>> session_response =
        server_session_->GetOutgoingMessage();
    if (!session_response.ok()) {
      return ToGrpcStatus(session_response.status());
    }
    // There wasn't exactly 1:1 mapping between the handshake requests and
    // responses. For example, when the client sends the handshake request with
    // the attestation binding, the server will accept the request and open a
    // session, and no response will be sent back. We will return an empty
    // ComputationResponse in this case.
    if (session_response->has_value()) {
      response->mutable_result()->PackFrom(session_response->value());
    }
    return grpc::Status::OK;
  }

  // Handle the computation request.
  auto plaintext_request = DecryptRequest(session_request);
  if (!plaintext_request.ok()) {
    return ToGrpcStatus(plaintext_request.status());
  }
  TffSessionConfig comp_request;
  bool parse_success =
      comp_request.ParseFromString(plaintext_request->plaintext());
  if (!parse_success) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Could not parse TffSessionConfig from the request.");
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
  // TODO: b/378243349 - Encrypt the result into a blob with the
  // output_access_policy_node_id set in the request.
  PlaintextMessage plaintext_result;
  plaintext_result.set_plaintext(call_result.SerializeAsString());
  auto session_response = EncryptResult(plaintext_result);
  if (!session_response.ok()) {
    return ToGrpcStatus(session_response.status());
  }
  response->mutable_result()->PackFrom(*session_response);

  return grpc::Status::OK;
}

}  // namespace confidential_federated_compute::program_worker
