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

#include "pi_client.h"

#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "cc/ffi/bytes_bindings.h"
#include "cc/ffi/bytes_view.h"
#include "cc/oak_session/client_session.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/oak_session_bindings.h"
#include "cc/oak_session/server_session.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/com/google/android/as/oss/privateinference/api/private_aratea_service.pb.h"
#include "src/com/google/android/as/oss/privateinference/service/api/private_inference.pb.h"

namespace confidential_federated_compute::private_inference {
namespace {

namespace ffi_bindings = ::oak::ffi::bindings;
namespace bindings = ::oak::session::bindings;

using ::oak::session::AttestationType;
using ::oak::session::ClientSession;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;

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
    LOG(FATAL) << "Failed to create attester";
  }
  ffi_bindings::free_rust_bytes_contents(fake_evidence);

  auto fake_endorsements =
      bindings::new_fake_endorsements(ffi_bindings::BytesView(kFakePlatform));
  auto endorser =
      bindings::new_simple_endorser(ffi_bindings::BytesView(fake_endorsements));
  if (endorser.error != nullptr) {
    LOG(FATAL) << "Failed to create attester";
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

TEST(PiClientTest, RequestResponseDecryptionSucceeds) {
  auto client_session_or = ClientSession::Create(TestConfigAttestedNNClient());
  ASSERT_TRUE(client_session_or.ok());
  auto client_session = std::move(client_session_or.value());

  auto server_session_or =
      oak::session::ServerSession::Create(TestConfigAttestedNNServer());
  ASSERT_TRUE(server_session_or.ok());
  auto server_session = std::move(server_session_or.value());

  // Perform handshake
  while (!client_session->IsOpen() || !server_session->IsOpen()) {
    auto client_msg = client_session->GetOutgoingMessage();
    if (client_msg.ok() && client_msg->has_value()) {
      server_session->PutIncomingMessage(client_msg->value());
    }
    auto server_msg = server_session->GetOutgoingMessage();
    if (server_msg.ok() && server_msg->has_value()) {
      client_session->PutIncomingMessage(server_msg->value());
    }
  }

  ASSERT_TRUE(client_session->IsOpen());
  ASSERT_TRUE(server_session->IsOpen());

  // Client sends request
  std::string plaintext_request = "test prompt";
  auto write_status = client_session->Write(plaintext_request);
  ASSERT_TRUE(write_status.ok());

  auto encrypted_request = client_session->GetOutgoingMessage();
  ASSERT_TRUE(encrypted_request.ok());
  ASSERT_TRUE(encrypted_request->has_value());

  // Server receives and decrypts
  server_session->PutIncomingMessage(encrypted_request->value());
  auto decrypted_request_or = server_session->ReadToRustBytes();
  ASSERT_TRUE(decrypted_request_or.ok());
  ASSERT_TRUE(decrypted_request_or->has_value());
  std::string decrypted_bytes =
      static_cast<std::string>(decrypted_request_or->value());
  EXPECT_EQ(decrypted_bytes, plaintext_request);

  // Server sends response
  ::mdi::privatearatea::PcsPrivateArateaResponse proto_response;
  proto_response.set_request_id(42);
  auto* candidate =
      proto_response.mutable_generate_content_response()->add_candidates();
  candidate->mutable_content()->add_parts()->set_text("generated answer");
  candidate->set_finish_reason(::mdi::privatearatea::PcsCandidate::STOP);
  std::string serialized_response = proto_response.SerializeAsString();

  auto server_write_status = server_session->Write(serialized_response);
  ASSERT_TRUE(server_write_status.ok());

  auto encrypted_response = server_session->GetOutgoingMessage();
  ASSERT_TRUE(encrypted_response.ok());
  ASSERT_TRUE(encrypted_response->has_value());

  // Client receives and decrypts
  client_session->PutIncomingMessage(encrypted_response->value());
  auto decrypted_response_or = client_session->ReadToRustBytes();
  ASSERT_TRUE(decrypted_response_or.ok());
  ASSERT_TRUE(decrypted_response_or->has_value());

  std::string payload =
      static_cast<std::string>(decrypted_response_or->value());

  ::mdi::privatearatea::PcsPrivateArateaResponse parsed_response;
  ASSERT_TRUE(parsed_response.ParseFromString(payload));
  EXPECT_EQ(parsed_response.request_id(), 42);
  EXPECT_EQ(parsed_response.generate_content_response()
                .candidates(0)
                .content()
                .parts(0)
                .text(),
            "generated answer");
  EXPECT_EQ(
      parsed_response.generate_content_response().candidates(0).finish_reason(),
      ::mdi::privatearatea::PcsCandidate::STOP);
}

TEST(PiClientTest, CreatePiClientFailsWhenServerIsUnreachable) {
  auto client_or = CreatePiClient("localhost:12345", 100);

  // CreatePiClient now performs the handshake, so it should fail if
  // unreachable.
  EXPECT_FALSE(client_or.ok());
  EXPECT_THAT(client_or.status().message(),
              testing::AnyOf(
                  testing::HasSubstr("ExchangeHandshakeMessages failed"),
                  testing::HasSubstr("Failed to open verification keys file"),
                  testing::HasSubstr("ClientSession::Create failed")));
}

TEST(PiClientTest, CreatePiClientDoesNotFailOnInvalidFeatureId) {
  auto client_or = CreatePiClient("localhost:12345", 999);
  EXPECT_FALSE(client_or.ok());
  EXPECT_NE(client_or.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace confidential_federated_compute::private_inference
