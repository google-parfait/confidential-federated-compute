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

#include "learning_containers/program_executor_tee/program_context/cc/noise_client_session.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "cc/oak_session/client_session.h"
#include "cc/oak_session/config.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"

using ::fcp::confidentialcompute::outgoing::ComputationDelegation;
using ::fcp::confidentialcompute::outgoing::ComputationRequest;
using ::fcp::confidentialcompute::outgoing::ComputationResponse;
using ::oak::session::ClientSession;
using ::oak::session::SessionConfig;
using ::oak::session::v1::PlaintextMessage;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

namespace confidential_federated_compute::program_executor_tee {

absl::Status NoiseClientSession::OpenSession() {
  while (!client_session_->IsOpen()) {
    FCP_ASSIGN_OR_RETURN(auto init_request,
                         client_session_->GetOutgoingMessage());
    if (!init_request.has_value()) {
      return absl::InternalError("init_request doesn't have value.");
    }
    ComputationRequest request;
    request.set_worker_bns(worker_bns_);
    request.mutable_computation()->PackFrom(init_request.value());
    ComputationResponse response;
    {
      grpc::ClientContext client_context;
      auto status = stub_->Execute(&client_context, request, &response);
      if (!status.ok()) {
        return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                            status.error_message());
      }
    }
    if (response.has_result()) {
      SessionResponse init_response;
      if (!response.result().UnpackTo(&init_response)) {
        return absl::InternalError(
            "Failed to unpack response to init_response.");
      }
      FCP_RETURN_IF_ERROR(client_session_->PutIncomingMessage(init_response));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<PlaintextMessage> NoiseClientSession::DelegateComputation(
    const PlaintextMessage& plaintext_request) {
  // Open the session if it is not already open. This is needed before sending
  // any actual computation request.
  if (!client_session_->IsOpen()) {
    FCP_RETURN_IF_ERROR(OpenSession());
  } else {
    LOG(INFO) << "Session is already open for worker: " << worker_bns_;
  }
  FCP_ASSIGN_OR_RETURN(auto session_request, EncryptRequest(plaintext_request));
  ComputationRequest request;
  request.mutable_computation()->PackFrom(session_request);
  request.set_worker_bns(worker_bns_);
  ComputationResponse response;
  {
    grpc::ClientContext client_context;
    auto status = stub_->Execute(&client_context, request, &response);
    if (!status.ok()) {
      return absl::Status(static_cast<absl::StatusCode>(status.error_code()),
                          status.error_message());
    }
  }
  if (!response.has_result()) {
    return absl::InternalError("Response doesn't have result.");
  }
  SessionResponse session_response;
  if (!response.result().UnpackTo(&session_response)) {
    return absl::InternalError(
        "Failed to unpack response to session_response.");
  }
  FCP_ASSIGN_OR_RETURN(auto plaintext_response,
                       DecryptResponse(session_response));
  return plaintext_response;
}

absl::StatusOr<SessionRequest> NoiseClientSession::EncryptRequest(
    const PlaintextMessage& plaintext_request) {
  FCP_RETURN_IF_ERROR(client_session_->Write(plaintext_request));
  FCP_ASSIGN_OR_RETURN(auto session_request,
                       client_session_->GetOutgoingMessage());
  if (!session_request.has_value()) {
    return absl::InvalidArgumentError(
        "Could not generate SessionRequest for the plaintext request.");
  }
  return session_request.value();
}

absl::StatusOr<PlaintextMessage> NoiseClientSession::DecryptResponse(
    const SessionResponse& session_response) {
  FCP_RETURN_IF_ERROR(client_session_->PutIncomingMessage(session_response));
  FCP_ASSIGN_OR_RETURN(auto plaintext_response, client_session_->Read());
  if (!plaintext_response.has_value()) {
    return absl::InvalidArgumentError(
        "Could not read plaintext message from the response.");
  }
  return plaintext_response.value();
}

}  // namespace confidential_federated_compute::program_executor_tee
