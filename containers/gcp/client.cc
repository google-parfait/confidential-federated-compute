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

#include "client.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "attestation_policy.pb.h"
#include "attestation_token_verifier.h"
#include "cc/containers/sdk/orchestrator_client.h"
#include "cc/oak_session/client_session.h"
#include "client_session_config.h"
#include "google/protobuf/text_format.h"
#include "grpcpp/grpcpp.h"
#include "oak_session_utils.h"
#include "proto/services/session_v1_service.grpc.pb.h"
#include "proto_parsing_utils.h"

namespace confidential_federated_compute::gcp {
namespace {

using ::oak::containers::sdk::OrchestratorClient;
using ::oak::services::OakSessionV1Service;
using ::oak::session::ClientSession;
using ::oak::session::SessionConfig;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

class ClientImpl : public Client {
 public:
  explicit ClientImpl(
      std::unique_ptr<AttestationTokenVerifier> verifier,
      std::unique_ptr<ClientSession> session,
      std::shared_ptr<grpc::Channel> channel,
      std::unique_ptr<oak::services::OakSessionV1Service::Stub> stub,
      std::unique_ptr<grpc::ClientContext> context,
      std::unique_ptr<grpc::ClientReaderWriter<
          oak::session::v1::SessionRequest, oak::session::v1::SessionResponse>>
          stream)
      : verifier_(std::move(verifier)),
        session_(std::move(session)),
        channel_(channel),
        stub_(std::move(stub)),
        context_(std::move(context)),
        stream_(std::move(stream)) {}

  virtual ~ClientImpl() override {
    LOG(INFO) << "Closing stream to GCP.";

    stream_->WritesDone();
    grpc::Status status = stream_->Finish();
    if (!status.ok()) {
      LOG(ERROR) << "Error while closing the GCP stream: "
                 << status.error_code() << ": " << status.error_message();
    }
  }

  virtual absl::StatusOr<std::string> Invoke(std::string request) override {
    absl::Status write_status = session_->Write(request);
    if (!write_status.ok()) {
      return absl::InternalError(absl::StrCat("Failed to write to session: ",
                                              write_status.ToString()));
    }

    absl::StatusOr<bool> pump_status =
        PumpOutgoingMessages(session_.get(), stream_.get());
    if (!pump_status.ok()) {
      return absl::InternalError(absl::StrCat("PumpOutgoingMessages failed: ",
                                              pump_status.status().ToString()));
    }

    while (true) {
      SessionResponse session_response;
      if (!stream_->Read(&session_response)) {
        return absl::InternalError(
            "Server closed stream while waiting for application reply.");
      }

      absl::Status put_status = session_->PutIncomingMessage(session_response);
      if (!pump_status.ok()) {
        return absl::InternalError(absl::StrCat("PutIncomingMessage failed:  ",
                                                put_status.ToString()));
      }

      auto decrypted_message = session_->ReadToRustBytes();
      if (!decrypted_message.ok()) {
        return absl::InternalError(
            absl::StrCat("Failed to read from session: ",
                         decrypted_message.status().ToString()));
      }

      if (decrypted_message->has_value()) {
        std::string payload =
            static_cast<std::string>(decrypted_message->value());

        // Success! We received the batch response.
        return payload;
      }

      pump_status = PumpOutgoingMessages(session_.get(), stream_.get());
      if (!pump_status.ok()) {
        return absl::InternalError(absl::StrCat(
            "PumpOutgoingMessages failed: ", pump_status.status().ToString()));
      }
    }
    return absl::InternalError("Unreachable.");
  }

 private:
  std::unique_ptr<AttestationTokenVerifier> verifier_;
  std::unique_ptr<ClientSession> session_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<oak::services::OakSessionV1Service::Stub> stub_;
  std::unique_ptr<grpc::ClientContext> context_;
  std::unique_ptr<grpc::ClientReaderWriter<oak::session::v1::SessionRequest,
                                           oak::session::v1::SessionResponse>>
      stream_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Client>> CreateClient(
    std::string server_address,
    std::unique_ptr<AttestationTokenVerifier> attestation_token_verifier) {
  if (!attestation_token_verifier) {
    return absl::InternalError("Attestation verifier cannot be null.");
  }

  // Initialize Session & Connect
  LOG(INFO) << "Initializing Oak ClientSession via Rust.";

  // We use FFI here because the Oak ClientSession is implemented in Rust.
  // We pass the C++ verifier object and a callback function so that Rust can
  // delegate the actual JWT verification back to our C++ implementation.
  SessionConfig* client_config = create_client_session_config(
      static_cast<void*>(attestation_token_verifier.get()), &verify_jwt_f);

  absl::StatusOr<std::unique_ptr<ClientSession>> client_session_or =
      ClientSession::Create(client_config);
  if (!client_session_or.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to create ClientSession: ",
                     client_session_or.status().ToString()));
  }

  LOG(INFO) << "Connecting to server at: " << server_address;

  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  auto stub = OakSessionV1Service::NewStub(channel);
  std::unique_ptr<grpc::ClientContext> client_context =
      std::make_unique<grpc::ClientContext>();
  auto stream = stub->Stream(client_context.get());

  // Execute Flow: Handshake, send message, receive response.
  // The handshake will trigger the Rust session to call our C++ verification
  // callback.
  absl::Status handshake_status =
      ExchangeHandshakeMessages(client_session_or->get(), stream.get());
  if (!handshake_status.ok()) {
    return handshake_status;
  }

  return std::make_unique<ClientImpl>(
      std::move(attestation_token_verifier), std::move(*client_session_or),
      channel, std::move(stub), std::move(client_context), std::move(stream));
}

}  // namespace confidential_federated_compute::gcp
