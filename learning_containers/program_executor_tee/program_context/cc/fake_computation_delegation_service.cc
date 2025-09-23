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

#include "program_executor_tee/program_context/cc/fake_computation_delegation_service.h"

#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "cc/ffi/bytes_bindings.h"
#include "cc/ffi/bytes_view.h"
#include "cc/ffi/error_bindings.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "cc/oak_session/server_session.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/tff_execution_helper.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "proto/session/session.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::TffSessionConfig;
using ::fcp::confidentialcompute::outgoing::ComputationRequest;
using ::fcp::confidentialcompute::outgoing::ComputationResponse;
using ::grpc::ServerContext;
using ::grpc::Status;
using ::oak::session::AttestationType;
using ::oak::session::HandshakeType;
using ::oak::session::ServerSession;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;
using ::oak::session::v1::PlaintextMessage;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

constexpr absl::string_view kFakeAttesterId = "fake_attester";
constexpr absl::string_view kFakeEvent = "fake event";
constexpr absl::string_view kFakePlatform = "fake platform";

namespace {

SessionConfig* TestConfigAttestedNNServer() {
  auto signing_key = bindings::new_random_signing_key();
  auto verifying_bytes = bindings::signing_key_verifying_key_bytes(signing_key);

  auto fake_evidence =
      bindings::new_fake_evidence(ffi_bindings::BytesView(verifying_bytes),
                                  ffi_bindings::BytesView(kFakeEvent));
  ffi_bindings::free_rust_bytes_contents(verifying_bytes);
  auto attester =
      bindings::new_simple_attester(ffi_bindings::BytesView(fake_evidence));
  if (attester.error != nullptr) {
    LOG(FATAL) << "Failed to create attester:"
               << ffi_bindings::ErrorIntoStatus(attester.error);
  }
  ffi_bindings::free_rust_bytes_contents(fake_evidence);

  auto fake_endorsements =
      bindings::new_fake_endorsements(ffi_bindings::BytesView(kFakePlatform));
  auto endorser =
      bindings::new_simple_endorser(ffi_bindings::BytesView(fake_endorsements));
  if (endorser.error != nullptr) {
    LOG(FATAL) << "Failed to create attester:" << attester.error;
  }

  ffi_bindings::free_rust_bytes_contents(fake_endorsements);

  auto builder = SessionConfigBuilder(AttestationType::kSelfUnidirectional,
                                      HandshakeType::kNoiseNN)
                     .AddSelfAttester(kFakeAttesterId, attester.result)
                     .AddSelfEndorser(kFakeAttesterId, endorser.result)
                     .AddSessionBinder(kFakeAttesterId, signing_key);

  bindings::free_signing_key(signing_key);

  return builder.Build();
}

}  // namespace

FakeComputationDelegationService::FakeComputationDelegationService(
    std::vector<std::string> worker_bns,
    std::function<
        absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
        leaf_executor_factory)
    : leaf_executor_factory_(leaf_executor_factory) {
  for (const auto& worker_bns : worker_bns) {
    auto server_session = ServerSession::Create(TestConfigAttestedNNServer());
    CHECK_OK(server_session);
    server_sessions_[worker_bns] = std::move(server_session.value());
  }
}

absl::StatusOr<PlaintextMessage>
FakeComputationDelegationService::DecryptRequest(
    const SessionRequest& session_request, std::string worker_bns) {
  FCP_RETURN_IF_ERROR(
      server_sessions_[worker_bns]->PutIncomingMessage(session_request));
  FCP_ASSIGN_OR_RETURN(auto plaintext_request,
                       server_sessions_[worker_bns]->Read());
  if (!plaintext_request.has_value()) {
    return absl::InvalidArgumentError(
        "Could not read plaintext message from the request.");
  }
  return plaintext_request.value();
}

absl::StatusOr<SessionResponse> FakeComputationDelegationService::EncryptResult(
    const PlaintextMessage& plaintext_response, std::string worker_bns) {
  FCP_RETURN_IF_ERROR(server_sessions_[worker_bns]->Write(plaintext_response));
  FCP_ASSIGN_OR_RETURN(auto session_response,
                       server_sessions_[worker_bns]->GetOutgoingMessage());
  if (!session_response.has_value()) {
    return absl::InvalidArgumentError(
        "Could not generate SessionResponse for the request.");
  }
  return session_response.value();
}

absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
FakeComputationDelegationService::InitializeTffExecutor(
    const TffSessionConfig& comp_request) {
  auto leaf_executor_fn = [this]()
      -> absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>> {
    FCP_ASSIGN_OR_RETURN(auto executor, leaf_executor_factory_());
    return tensorflow_federated::CreateReferenceResolvingExecutor(executor);
  };
  tensorflow_federated::CardinalityMap cardinality_map;
  cardinality_map[tensorflow_federated::kClientsUri] =
      comp_request.num_clients();
  FCP_ASSIGN_OR_RETURN(auto server_child, leaf_executor_fn());
  FCP_ASSIGN_OR_RETURN(auto client_child, leaf_executor_fn());
  FCP_ASSIGN_OR_RETURN(auto federating_executor,
                       tensorflow_federated::CreateFederatingExecutor(
                           /*server_child=*/server_child,
                           /*client_child=*/client_child, cardinality_map));
  return tensorflow_federated::CreateReferenceResolvingExecutor(
      federating_executor);
}

Status FakeComputationDelegationService::Execute(
    ServerContext* context, const ComputationRequest* request,
    ComputationResponse* response) {
  absl::MutexLock lock(&mutex_);
  std::string worker_bns = request->worker_bns();
  SessionRequest session_request;
  if (!request->computation().UnpackTo(&session_request)) {
    return grpc::Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        "ComputationRequest cannot be unpacked to noise SessionRequest.");
  }

  // Perform handshake until the channel is open.
  if (!server_sessions_[worker_bns]->IsOpen()) {
    auto put_incoming_message_status =
        server_sessions_[worker_bns]->PutIncomingMessage(session_request);
    if (!put_incoming_message_status.ok()) {
      return ToGrpcStatus(put_incoming_message_status);
    }
    absl::StatusOr<std::optional<SessionResponse>> session_response =
        server_sessions_[worker_bns]->GetOutgoingMessage();
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
  auto plaintext_request = DecryptRequest(session_request, worker_bns);
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

  PlaintextMessage plaintext_result;
  plaintext_result.set_plaintext(call_result.SerializeAsString());
  auto session_response = EncryptResult(plaintext_result, worker_bns);
  if (!session_response.ok()) {
    return ToGrpcStatus(session_response.status());
  }
  response->mutable_result()->PackFrom(*session_response);

  return grpc::Status::OK;
}

}  // namespace confidential_federated_compute::program_executor_tee
