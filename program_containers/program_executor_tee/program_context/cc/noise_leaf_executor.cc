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
#include "program_executor_tee/program_context/cc/noise_leaf_executor.h"

#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "fcp/base/status_converters.h"
#include "program_executor_tee/proto/executor_wrapper.pb.h"
#include "proto/session/session.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_service.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ComputationRequest;
using ::fcp::confidentialcompute::ComputationResponse;
using ::grpc::Status;
using ::oak::session::v1::PlaintextMessage;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;
using ::tensorflow_federated::CardinalityMap;
using ::tensorflow_federated::CreateFederatingExecutor;
using ::tensorflow_federated::CreateReferenceResolvingExecutor;
using ::tensorflow_federated::Executor;

namespace {

std::function<absl::StatusOr<std::shared_ptr<Executor>>(
    const CardinalityMap& cardinality_map)>
CreateLeafExecutorFactory(
    std::function<absl::StatusOr<std::shared_ptr<Executor>>()>
        leaf_executor_factory) {
  return [leaf_executor_factory](const CardinalityMap& cardinality_map)
             -> absl::StatusOr<std::shared_ptr<Executor>> {
    FCP_ASSIGN_OR_RETURN(auto server_child, leaf_executor_factory());
    FCP_ASSIGN_OR_RETURN(auto client_child, leaf_executor_factory());
    FCP_ASSIGN_OR_RETURN(
        std::shared_ptr<Executor> federating_executor,
        CreateFederatingExecutor(CreateReferenceResolvingExecutor(server_child),
                                 CreateReferenceResolvingExecutor(client_child),
                                 cardinality_map));
    auto ref = CreateReferenceResolvingExecutor(federating_executor);
    return ref;
  };
}

bool IsHandshakeRequest(const SessionRequest& session_request) {
  return session_request.has_attest_request() ||
         session_request.has_handshake_request();
}

}  // namespace

NoiseLeafExecutor::NoiseLeafExecutor(
    std::function<oak::session::SessionConfig*()> session_config_fn,
    std::function<absl::StatusOr<std::shared_ptr<Executor>>()>
        leaf_executor_factory)
    : executor_service_(std::make_unique<tensorflow_federated::ExecutorService>(
          CreateLeafExecutorFactory(leaf_executor_factory))),
      session_config_fn_(session_config_fn) {}

absl::StatusOr<PlaintextMessage> NoiseLeafExecutor::DecryptRequest(
    const SessionRequest& session_request) {
  FCP_RETURN_IF_ERROR(server_session_->PutIncomingMessage(session_request));
  FCP_ASSIGN_OR_RETURN(auto plaintext_request, server_session_->Read());
  if (!plaintext_request.has_value()) {
    return absl::InvalidArgumentError(
        "Could not read plaintext message from the request.");
  }
  return plaintext_request.value();
}

absl::StatusOr<SessionResponse> NoiseLeafExecutor::EncryptResult(
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

absl::StatusOr<std::optional<SessionResponse>>
NoiseLeafExecutor::HandleHandshakeRequest(
    const SessionRequest& session_request) {
  if (server_session_ == nullptr || server_session_->IsOpen()) {
    // Create a new session in two cases:
    // (1) The session doesn't exist.
    // (2) If a handshake request is received on an already-open session, close
    // the old session and create a new one. This handles cases where a client
    // restarts and reconnects.
    auto* session_config = session_config_fn_();
    if (session_config == nullptr) {
      return absl::InvalidArgumentError(
          "The session config function returned a null pointer.");
    }
    FCP_ASSIGN_OR_RETURN(server_session_,
                         oak::session::ServerSession::Create(session_config));
  }

  auto put_incoming_message_status =
      server_session_->PutIncomingMessage(session_request);
  if (!put_incoming_message_status.ok()) {
    server_session_ = nullptr;
    return put_incoming_message_status;
  }
  absl::StatusOr<std::optional<SessionResponse>> session_response =
      server_session_->GetOutgoingMessage();
  if (!session_response.ok()) {
    server_session_ = nullptr;
    return session_response.status();
  }
  return session_response.value();
}

grpc::Status NoiseLeafExecutor::Execute(const ComputationRequest* request,
                                        ComputationResponse* response) {
  SessionRequest session_request;
  if (!request->computation().UnpackTo(&session_request)) {
    return grpc::Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        "ComputationRequest cannot be unpacked to noise SessionRequest.");
  }

  // Handle the handshake request.
  if (IsHandshakeRequest(session_request)) {
    auto session_response = HandleHandshakeRequest(session_request);
    if (!session_response.ok()) {
      return ToGrpcStatus(session_response.status());
    }
    // There wasn't exactly 1:1 mapping between the handshake requests and
    // responses. For example, when the client sends the handshake request
    // with the attestation binding, the server will accept the request and open
    // a session, and no response will be sent back. We will return an empty
    // ComputationResponse in this case.
    if (session_response->has_value()) {
      response->mutable_result()->PackFrom(session_response->value());
    }
    return grpc::Status::OK;
  }

  // Handle the remote execution request.
  if (server_session_ == nullptr) {
    return grpc::Status(
        grpc::StatusCode::FAILED_PRECONDITION,
        "Server session is not created yet, but received computation request.");
  }
  auto plaintext_request = DecryptRequest(session_request);
  if (!plaintext_request.ok()) {
    return ToGrpcStatus(plaintext_request.status());
  }
  executor_wrapper::ExecutorGroupRequest executor_group_request;
  bool parse_success =
      executor_group_request.ParseFromString(plaintext_request->plaintext());
  if (!parse_success) {
    return grpc::Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        "Could not parse ExecutorGroupRequest from the request.");
  }
  executor_wrapper::ExecutorGroupResponse executor_group_response;
  switch (executor_group_request.request_case()) {
    case executor_wrapper::ExecutorGroupRequest::kGetExecutorRequest: {
      tensorflow_federated::v0::GetExecutorResponse get_executor_response;
      executor_service_->GetExecutor(
          nullptr, &executor_group_request.get_executor_request(),
          &get_executor_response);
      *executor_group_response.mutable_get_executor_response() =
          std::move(get_executor_response);
      break;
    }
    case executor_wrapper::ExecutorGroupRequest::kCreateValueRequest: {
      tensorflow_federated::v0::CreateValueResponse create_value_response;
      executor_service_->CreateValue(
          nullptr, &executor_group_request.create_value_request(),
          &create_value_response);
      *executor_group_response.mutable_create_value_response() =
          std::move(create_value_response);
      break;
    }
    case executor_wrapper::ExecutorGroupRequest::kCreateCallRequest: {
      tensorflow_federated::v0::CreateCallResponse create_call_response;
      executor_service_->CreateCall(
          nullptr, &executor_group_request.create_call_request(),
          &create_call_response);
      *executor_group_response.mutable_create_call_response() =
          std::move(create_call_response);
      break;
    }
    case executor_wrapper::ExecutorGroupRequest::kCreateStructRequest: {
      tensorflow_federated::v0::CreateStructResponse create_struct_response;
      executor_service_->CreateStruct(
          nullptr, &executor_group_request.create_struct_request(),
          &create_struct_response);
      *executor_group_response.mutable_create_struct_response() =
          std::move(create_struct_response);
      break;
    }
    case executor_wrapper::ExecutorGroupRequest::kCreateSelectionRequest: {
      tensorflow_federated::v0::CreateSelectionResponse
          create_selection_response;
      executor_service_->CreateSelection(
          nullptr, &executor_group_request.create_selection_request(),
          &create_selection_response);
      *executor_group_response.mutable_create_selection_response() =
          std::move(create_selection_response);
      break;
    }
    case executor_wrapper::ExecutorGroupRequest::kComputeRequest: {
      tensorflow_federated::v0::ComputeResponse compute_response;
      executor_service_->Compute(nullptr,
                                 &executor_group_request.compute_request(),
                                 &compute_response);
      *executor_group_response.mutable_compute_response() =
          std::move(compute_response);
      break;
    }
    case executor_wrapper::ExecutorGroupRequest::kDisposeRequest: {
      tensorflow_federated::v0::DisposeResponse dispose_response;
      executor_service_->Dispose(nullptr,
                                 &executor_group_request.dispose_request(),
                                 &dispose_response);
      *executor_group_response.mutable_dispose_response() =
          std::move(dispose_response);
      break;
    }
    case executor_wrapper::ExecutorGroupRequest::kDisposeExecutorRequest: {
      tensorflow_federated::v0::DisposeExecutorResponse
          disponse_executor_response;
      executor_service_->DisposeExecutor(
          nullptr, &executor_group_request.dispose_executor_request(),
          &disponse_executor_response);
      *executor_group_response.mutable_dispose_executor_response() =
          std::move(disponse_executor_response);
      break;
    }
    case executor_wrapper::ExecutorGroupRequest::REQUEST_NOT_SET: {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "ExecutorGroupRequest missing request.");
    }
  }
  PlaintextMessage plaintext_result;
  plaintext_result.set_plaintext(executor_group_response.SerializeAsString());
  auto session_response = EncryptResult(plaintext_result);
  if (!session_response.ok()) {
    return ToGrpcStatus(session_response.status());
  }
  response->mutable_result()->PackFrom(*session_response);

  return grpc::Status::OK;
}

}  // namespace confidential_federated_compute::program_executor_tee
