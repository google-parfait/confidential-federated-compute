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

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "proto/private_aratea_service.pb.h"
#include "proto/verification_keys.pb.h"

// Oak includes
#include "proto/services/session_v1_service.grpc.pb.h"
// Assuming these paths based on provided code, might need adjustment.
#include "cc/oak_session/client_session.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/server_session.h"
#include "oak_session_utils.h"
#include "proto/services/session_v1_service.grpc.pb.h"

// Forward declaration of the Rust function.
extern "C" {
::oak::session::bindings::SessionConfigBuilder*
update_peer_unidirectional_session_config(
    ::oak::session::bindings::SessionConfigBuilder* builder,
    const char* tink_serialized_public_keyset_data,
    int tink_serialized_public_keyset_len);
}

constexpr absl::string_view kProdVerificationKeysPath =
    "private_inference/public_keys_prod_proto.binarypb";

namespace confidential_federated_compute::private_inference {

namespace {

using ::oak::session::ClientSession;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;
using ::private_inference::proto::GenerateContentRequest;
using ::private_inference::proto::PrivateArateaRequest;
using ::private_inference::proto::VerificationKeys;

class PiClientImpl : public PiClient {
 public:
  explicit PiClientImpl(std::string server_address,
                        ::private_inference::proto::FeatureName feature_name)
      : server_address_(std::move(server_address)),
        feature_name_(feature_name) {}

  absl::Status Initialize() {
    // 1. Establish channel
    LOG(INFO) << "Creating channel to PI server: " << server_address_;
    channel_ = grpc::CreateChannel(server_address_,
                                   grpc::InsecureChannelCredentials());
    stub_ = oak::services::OakSessionV1Service::NewStub(channel_);

    stream_ = stub_->Stream(&context_);

    // 2. Configure session (Attestation)
    VerificationKeys verification_keys;
    std::ifstream input(std::string(kProdVerificationKeysPath),
                        std::ios::binary);
    if (!input) {
      return absl::InternalError(absl::StrCat(
          "PiClientImpl::Initialize: Failed to open verification keys file: ",
          kProdVerificationKeysPath));
    }
    if (!verification_keys.ParseFromIstream(&input)) {
      return absl::InternalError(
          "PiClientImpl::Initialize: Failed to parse verification keys.");
    }
    auto builder = ::oak::session::SessionConfigBuilder(
        oak::session::AttestationType::kPeerUnidirectional,
        oak::session::HandshakeType::kNoiseNN);

    auto status = builder.UpdateRaw(
        [keys = verification_keys.tink_serialized_public_keyset()](
            ::oak::session::SessionConfigBuilderHolder raw_builder) {
          auto new_builder = update_peer_unidirectional_session_config(
              raw_builder.release(), keys.data(), keys.length());
          return ::oak::session::SessionConfigBuilderHolder(new_builder);
        });

    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat(
              "PiClientImpl::Initialize: Failed to build session config: ",
              status.message()));
    }
    LOG(INFO) << "Session config built.";
    auto session_config = builder.Build();
    auto session_or = ::oak::session::ClientSession::Create(session_config);
    if (!session_or.ok()) {
      return absl::Status(
          session_or.status().code(),
          absl::StrCat(
              "PiClientImpl::Initialize: ClientSession::Create failed: ",
              session_or.status().message()));
    }
    session_ = std::move(session_or.value());

    auto handshake_status =
        ExchangeHandshakeMessages(session_.get(), stream_.get());
    if (!handshake_status.ok()) {
      return absl::Status(
          handshake_status.code(),
          absl::StrCat(
              "PiClientImpl::Initialize: ExchangeHandshakeMessages failed: ",
              handshake_status.message()));
    }
    LOG(INFO) << "Handshake completed.";
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> Generate(const std::string& prompt) override {
    // Session is already established and handshake completed during
    // initialization.

    // 3. Send GenerateContentRequest wrapped in a PrivateArateaRequest.
    PrivateArateaRequest request;
    request.set_feature_name(feature_name_);

    GenerateContentRequest generate_request;
    generate_request.add_contents()->add_parts()->set_text(prompt);
    *request.mutable_generate_content_request() = generate_request;

    auto write_status = session_->Write(request.SerializeAsString());
    if (!write_status.ok()) {
      return absl::Status(
          write_status.code(),
          absl::StrCat("PiClientImpl::Generate: session::Write failed: ",
                       write_status.message()));
    }

    auto pump_status = PumpOutgoingMessages(session_.get(), stream_.get());
    if (!pump_status.ok()) {
      return absl::Status(
          pump_status.status().code(),
          absl::StrCat("PiClientImpl::Generate: PumpOutgoingMessages failed: ",
                       pump_status.status().message()));
    }

    while (true) {
      oak::session::v1::SessionResponse response;
      if (!stream_->Read(&response)) {
        return absl::InternalError(
            "PiClientImpl::Generate: Server closed stream while waiting for "
            "application reply.");
      }

      absl::Status put_status = session_->PutIncomingMessage(response);
      if (!put_status.ok()) {
        return absl::InternalError(
            absl::StrCat("PiClientImpl::Generate: PutIncomingMessage failed:  ",
                         put_status.ToString()));
      }

      auto decrypted_message = session_->ReadToRustBytes();
      if (!decrypted_message.ok()) {
        return absl::InternalError(absl::StrCat(
            "PiClientImpl::Generate: Failed to read from session: ",
            decrypted_message.status().ToString()));
      }

      if (decrypted_message->has_value()) {
        std::string payload =
            static_cast<std::string>(decrypted_message->value());

        // Success! We received the batch response.
        return payload;
      }
      auto pump_status2 = PumpOutgoingMessages(session_.get(), stream_.get());
      if (!pump_status2.ok()) {
        return absl::Status(
            pump_status2.status().code(),
            absl::StrCat("PiClientImpl::Generate: PumpOutgoingMessages (read "
                         "loop) failed: ",
                         pump_status2.status().message()));
      }
    }
  }

 private:
  std::string server_address_;
  ::private_inference::proto::FeatureName feature_name_;
  grpc::ClientContext context_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<oak::services::OakSessionV1Service::Stub> stub_;
  std::unique_ptr<grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream_;
  std::unique_ptr<ClientSession> session_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<PiClient>> CreatePiClient(
    std::string server_address,
    ::private_inference::proto::FeatureName feature_name) {
  auto client =
      std::make_unique<PiClientImpl>(std::move(server_address), feature_name);
  absl::Status status = client->Initialize();
  if (!status.ok()) {
    return status;
  }
  return client;
}

}  // namespace confidential_federated_compute::private_inference
