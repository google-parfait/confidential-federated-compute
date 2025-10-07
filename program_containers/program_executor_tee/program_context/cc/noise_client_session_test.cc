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

#include "program_executor_tee/program_context/cc/noise_client_session.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "cc/ffi/bytes_bindings.h"
#include "cc/ffi/bytes_view.h"
#include "cc/ffi/error_bindings.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "cc/oak_session/server_session.h"
#include "fcp/protos/confidentialcompute/computation_delegation.grpc.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation.pb.h"
#include "fcp/protos/confidentialcompute/computation_delegation_mock.grpc.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::fcp::confidentialcompute::outgoing::ComputationRequest;
using ::fcp::confidentialcompute::outgoing::ComputationResponse;
using ::fcp::confidentialcompute::outgoing::MockComputationDelegationStub;
using ::oak::session::AttestationType;
using ::oak::session::ClientSession;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;
using ::oak::session::v1::PlaintextMessage;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;
using ::testing::_;
using ::testing::Invoke;

constexpr char kWorkerBns[] = "/bns/test/worker";
constexpr absl::string_view kFakeAttesterId = "fake_attester";
constexpr absl::string_view kFakeEvent = "fake event";
constexpr absl::string_view kFakePlatform = "fake platform";

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

class NoiseClientSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto client_session = NoiseClientSession::Create(
        kWorkerBns, TestConfigAttestedNNClient(), mock_stub_.get());
    CHECK_OK(client_session);
    client_session_ = std::move(client_session.value());
    auto server_session =
        oak::session::ServerSession::Create(TestConfigAttestedNNServer());
    CHECK_OK(server_session);
    server_session_ = std::move(server_session.value());
    // Setup a mock to process the request with the server session.
    ON_CALL(*mock_stub_.get(), Execute(_, _, _))
        .WillByDefault(Invoke([this](grpc::ClientContext* context,
                                     const ComputationRequest& request,
                                     ComputationResponse* response) {
          return this->SimulateServerResponse(request, response);
        }));
  }
  // Simulate the server response using the server_session.
  grpc::Status SimulateServerResponse(const ComputationRequest& request,
                                      ComputationResponse* response) {
    SessionRequest session_request;
    CHECK_EQ(request.worker_bns(), kWorkerBns);
    if (!request.computation().UnpackTo(&session_request)) {
      return grpc::Status(grpc::StatusCode::INTERNAL,
                          "Failed to unpack request to SessionRequest.");
    }
    server_session_->PutIncomingMessage(session_request);
    if (server_session_->IsOpen()) {
      if (return_error_at_delegate_computation_) {
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "Failed to delegate computation.");
      }
      // Simply decrypt the request and encrypt it back.
      auto plaintext_request = server_session_->Read();
      if (!plaintext_request.ok() || !plaintext_request->has_value()) {
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "Failed to read plaintext request.");
      }
      server_session_->Write(plaintext_request->value());
    } else {
      if (return_error_at_open_session_) {
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "Failed to open session.");
      }
    }
    auto session_response = server_session_->GetOutgoingMessage();
    if (!session_response.ok()) {
      return grpc::Status(grpc::StatusCode::INTERNAL,
                          session_response.status().ToString());
    }
    if (session_response->has_value()) {
      response->mutable_result()->PackFrom(session_response->value());
    }

    return grpc::Status::OK;
  }

  std::unique_ptr<MockComputationDelegationStub> mock_stub_ =
      std::make_unique<MockComputationDelegationStub>();
  std::shared_ptr<NoiseClientSession> client_session_;
  std::unique_ptr<oak::session::ServerSession> server_session_;
  bool return_error_at_open_session_ = false;
  bool return_error_at_delegate_computation_ = false;
};

TEST_F(NoiseClientSessionTest, DelegateComputationSucceeds) {
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext("test");
  auto response = client_session_->DelegateComputation(plaintext_request);
  EXPECT_TRUE(response.ok());
  EXPECT_EQ(response->plaintext(), "test");
}

TEST_F(NoiseClientSessionTest, DelegateComputationFailsAtOpenSession) {
  return_error_at_open_session_ = true;
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext("test");
  auto response = client_session_->DelegateComputation(plaintext_request);
  EXPECT_FALSE(response.ok());
}

TEST_F(NoiseClientSessionTest, DelegateComputationFailsAtExecute) {
  return_error_at_delegate_computation_ = true;
  PlaintextMessage plaintext_request;
  plaintext_request.set_plaintext("test");
  auto response = client_session_->DelegateComputation(plaintext_request);
  EXPECT_FALSE(response.ok());
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee