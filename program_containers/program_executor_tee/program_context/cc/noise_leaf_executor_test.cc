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

#include "absl/status/status.h"
#include "cc/ffi/bytes_view.h"
#include "cc/oak_session/client_session.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "fcp/base/status_converters.h"
#include "federated_language_jax/executor/xla_executor.h"
#include "gtest/gtest.h"
#include "program_executor_tee/program_context/cc/test_helpers.h"
#include "program_executor_tee/proto/executor_wrapper.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"

namespace confidential_federated_compute::program_executor_tee {

extern "C" {
extern void init_tokio_runtime();
}

namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::fcp::base::FromGrpcStatus;
using ::fcp::confidentialcompute::ComputationRequest;
using ::fcp::confidentialcompute::ComputationResponse;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::oak::session::AttestationType;
using ::oak::session::ClientSession;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;
using ::oak::session::v1::PlaintextMessage;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;
using ::testing::Test;

constexpr absl::string_view kFakeAttesterId = "fake_attester";
constexpr absl::string_view kFakeEvent = "fake event";
constexpr absl::string_view kFakePlatform = "fake platform";
constexpr int kNumClients = 5;

absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>
CreateExecutor() {
  return federated_language_jax::CreateXLAExecutor();
}

SessionConfig* TestConfigAttestedNNClient() {
  auto verifier = bindings::new_fake_attestation_verifier(
      ffi_bindings::BytesView(kFakeEvent),
      ffi_bindings::BytesView(kFakePlatform));

  return SessionConfigBuilder(AttestationType::kPeerUnidirectional,
                              HandshakeType::kNoiseNN)
      .AddPeerVerifier(kFakeAttesterId, verifier)
      .Build();
}

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

SessionConfig* NullSessionConfigFn() { return nullptr; }

class NoiseLeafExecutorTest : public Test {
 public:
  NoiseLeafExecutorTest() {
    init_tokio_runtime();
    auto executor =
        NoiseLeafExecutor::Create(TestConfigAttestedNNServer, CreateExecutor);
    CHECK_OK(executor);
    executor_ = std::move(executor.value());
  }

  absl::StatusOr<std::unique_ptr<ClientSession>>
  CreateClientSessionAndDoHandshake() {
    FCP_ASSIGN_OR_RETURN(auto client_session,
                         ClientSession::Create(TestConfigAttestedNNClient()));

    while (!client_session->IsOpen()) {
      FCP_ASSIGN_OR_RETURN(auto init_request,
                           client_session->GetOutgoingMessage());
      if (!init_request.has_value()) {
        return absl::InternalError("init_request doesn't have value.");
      }
      ComputationRequest request;
      request.mutable_computation()->PackFrom(init_request.value());
      ComputationResponse response;
      {
        grpc::ClientContext client_context;
        auto status = executor_->Execute(&request, &response);
        if (!status.ok()) {
          return absl::InternalError("Failed to execute request on server.");
        }
      }
      if (response.has_result()) {
        SessionResponse init_response;
        if (!response.result().UnpackTo(&init_response)) {
          return absl::InternalError(
              "Failed to unpack response to init_response.");
        }
        FCP_RETURN_IF_ERROR(client_session->PutIncomingMessage(init_response));
      }
    }

    // Return an open client session.
    return client_session;
  }

  absl::StatusOr<executor_wrapper::ExecutorGroupResponse>
  MakeExecutorGroupRequest(executor_wrapper::ExecutorGroupRequest request,
                           ClientSession* client_session) {
    PlaintextMessage plaintext_comp_request;
    plaintext_comp_request.set_plaintext(request.SerializeAsString());
    FCP_RETURN_IF_ERROR(client_session->Write(plaintext_comp_request));
    FCP_ASSIGN_OR_RETURN(std::optional<SessionRequest> comp_session_request,
                         client_session->GetOutgoingMessage());

    ComputationRequest comp_request;
    comp_request.mutable_computation()->PackFrom(*comp_session_request);
    ComputationResponse comp_response;
    FCP_RETURN_IF_ERROR(
        FromGrpcStatus(executor_->Execute(&comp_request, &comp_response)));

    SessionResponse comp_session_response;
    if (!comp_response.result().UnpackTo(&comp_session_response)) {
      return absl::InternalError("Failed to unpack to SessionResponse");
    }
    FCP_RETURN_IF_ERROR(
        client_session->PutIncomingMessage(comp_session_response));
    FCP_ASSIGN_OR_RETURN(auto decrypted_comp_response, client_session->Read());
    executor_wrapper::ExecutorGroupResponse response;
    if (!response.ParseFromString(decrypted_comp_response->plaintext())) {
      return absl::InternalError("Failed to parse into ExecutorGroupResponse");
    }
    return response;
  }

 protected:
  std::unique_ptr<NoiseLeafExecutor> executor_;
};

TEST_F(NoiseLeafExecutorTest, ReconnectsAndCanContinue) {
  // Create a client session and do a handshake.
  auto client_session1_or = CreateClientSessionAndDoHandshake();
  ASSERT_TRUE(client_session1_or.ok());
  auto client_session1 = std::move(client_session1_or.value());

  // Use the session to ensure it's working.
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_get_executor_request() = GetExecutorRequest(kNumClients);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session1.get());
    ASSERT_TRUE(response.ok());
  }

  // Simulate a reconnect by sending a handshake request on an established
  // session. The server will close the existing session and open a new one.
  auto client_session2_or = ClientSession::Create(TestConfigAttestedNNClient());
  ASSERT_TRUE(client_session2_or.ok());
  auto client_session2 = std::move(client_session2_or.value());

  auto reconnect_req_or = client_session2->GetOutgoingMessage();
  ASSERT_TRUE(reconnect_req_or.ok());
  auto reconnect_req = reconnect_req_or.value();
  ASSERT_TRUE(reconnect_req.has_value());

  ComputationRequest request;
  request.mutable_computation()->PackFrom(*reconnect_req);
  ComputationResponse response;
  ASSERT_TRUE(executor_->Execute(&request, &response).ok());

  // Complete the handshake for the new session.
  if (response.has_result()) {
    SessionResponse init_response;
    ASSERT_TRUE(response.result().UnpackTo(&init_response));
    ASSERT_TRUE(client_session2->PutIncomingMessage(init_response).ok());
  }

  while (!client_session2->IsOpen()) {
    auto init_request_or = client_session2->GetOutgoingMessage();
    ASSERT_TRUE(init_request_or.ok());
    auto init_request = init_request_or.value();
    ASSERT_TRUE(init_request.has_value());
    ComputationRequest inner_request;
    inner_request.mutable_computation()->PackFrom(*init_request);
    ComputationResponse inner_response;
    ASSERT_TRUE(executor_->Execute(&inner_request, &inner_response).ok());
    if (inner_response.has_result()) {
      SessionResponse init_response;
      ASSERT_TRUE(inner_response.result().UnpackTo(&init_response));
      ASSERT_TRUE(client_session2->PutIncomingMessage(init_response).ok());
    }
  }

  // Make sure we can continue to use the service with the new session.
  // Create an executor.
  std::string executor_id;
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_get_executor_request() = GetExecutorRequest(kNumClients);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session2.get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_get_executor_response());
    executor_id = response->get_executor_response().executor().id();
    ASSERT_FALSE(executor_id.empty());
  }

  // Create the value 2.
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_create_value_request() =
        CreateIntValueRequest(executor_id, 2);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session2.get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_create_value_response());
    std::string value_two_ref =
        response->create_value_response().value_ref().id();
    ASSERT_FALSE(value_two_ref.empty());
  }
}

TEST_F(NoiseLeafExecutorTest, Success) {
  auto client_session = CreateClientSessionAndDoHandshake();
  ASSERT_TRUE(client_session.ok());

  // Create an executor.
  std::string executor_id;
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_get_executor_request() = GetExecutorRequest(kNumClients);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_get_executor_response());
    executor_id = response->get_executor_response().executor().id();
    ASSERT_FALSE(executor_id.empty());
  }

  // Create the value 2.
  std::string value_two_ref;
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_create_value_request() =
        CreateIntValueRequest(executor_id, 2);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_create_value_response());
    value_two_ref = response->create_value_response().value_ref().id();
    ASSERT_FALSE(value_two_ref.empty());
  }

  // Create the value 3.
  std::string value_three_ref;
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_create_value_request() =
        CreateIntValueRequest(executor_id, 3);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_create_value_response());
    value_three_ref = response->create_value_response().value_ref().id();
    ASSERT_FALSE(value_three_ref.empty());
  }

  // Create a struct from the values 2 and 3.
  std::string struct_ref;
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_create_struct_request() =
        CreateStructRequest(executor_id, {value_two_ref, value_three_ref});
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_create_struct_response());
    struct_ref = response->create_struct_response().value_ref().id();
    ASSERT_FALSE(struct_ref.empty());
  }

  // Create a selection of index 1 from the struct (will be the value 3).
  std::string selection_ref;
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_create_selection_request() =
        CreateSelectionRequest(executor_id, struct_ref, /*index=*/1);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_create_selection_response());
    selection_ref = response->create_selection_response().value_ref().id();
    ASSERT_FALSE(selection_ref.empty());
  }

  // Materialize the selection.
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_compute_request() =
        ComputeRequest(executor_id, selection_ref);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_compute_response());
    ASSERT_EQ(
        response->compute_response().value().array().int32_list().value(0), 3);
  }

  // Create a "federated_value_at_clients" intrinsic.
  std::string intrinsic_comp_ref;
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_create_value_request() =
        CreateIntrinsicValueRequest(executor_id, "federated_value_at_clients");
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_create_value_response());
    intrinsic_comp_ref = response->create_value_response().value_ref().id();
    ASSERT_FALSE(intrinsic_comp_ref.empty());
  }

  // Create a call for the "federated_value_at_clients" intrinsic on the
  // value 2.
  std::string call_ref;
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_create_call_request() =
        CreateCallRequest(executor_id, intrinsic_comp_ref, value_two_ref);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_create_call_response());
    call_ref = response->create_call_response().value_ref().id();
    ASSERT_FALSE(call_ref.empty());
  }

  // Materialize the call.
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_compute_request() = ComputeRequest(executor_id, call_ref);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_compute_response());
    auto federated_response = response->compute_response().value().federated();
    ASSERT_EQ(federated_response.type().placement().value().uri(), "clients");
    ASSERT_EQ(federated_response.value_size(), kNumClients);
  }

  // Dispose all refs.
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_dispose_request() = DisposeRequest(
        executor_id, {selection_ref, struct_ref, value_two_ref, value_three_ref,
                      intrinsic_comp_ref, call_ref});
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_dispose_response());
  }

  // Dispose the executor.
  {
    executor_wrapper::ExecutorGroupRequest request;
    *request.mutable_dispose_executor_request() =
        DisposeExecutorRequest(executor_id);
    absl::StatusOr<executor_wrapper::ExecutorGroupResponse> response =
        MakeExecutorGroupRequest(request, client_session.value().get());
    ASSERT_TRUE(response.ok());
    ASSERT_TRUE(response->has_dispose_executor_response());
  }
}

TEST_F(NoiseLeafExecutorTest, ReturnsErrorWhenSessionConfigFnReturnsNull) {
  auto executor_or =
      NoiseLeafExecutor::Create(NullSessionConfigFn, CreateExecutor);
  ASSERT_TRUE(executor_or.ok());
  auto executor = std::move(executor_or.value());

  // Create a client to initiate the handshake.
  auto client_session_or = ClientSession::Create(TestConfigAttestedNNClient());
  ASSERT_TRUE(client_session_or.ok());
  auto client_session = std::move(client_session_or.value());

  auto init_request_or = client_session->GetOutgoingMessage();
  ASSERT_TRUE(init_request_or.ok());
  ASSERT_TRUE(init_request_or->has_value());

  ComputationRequest request;
  request.mutable_computation()->PackFrom(init_request_or->value());
  ComputationResponse response;
  grpc::Status status = executor->Execute(&request, &response);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.error_message(),
            "The session config function returned a null pointer.");
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee
