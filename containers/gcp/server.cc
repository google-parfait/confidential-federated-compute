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

// gRPC server implementation using Oak Noise sessions for secure communication
// and GCP Confidential Space attestation.

#include "grpcpp/server.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "attestation_token_provider.h"
#include "cc/ffi/rust_bytes.h"
#include "cc/oak_session/server_session.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "oak_session_utils.h"
#include "proto/services/session_v1_service.grpc.pb.h"
#include "server.h"
#include "server_session_config.h"
#include "tink/config/tink_config.h"
#include "tink/signature/signature_config.h"

namespace confidential_federated_compute::gcp {
namespace {

using ::oak::services::OakSessionV1Service;
using ::oak::session::ServerSession;
using ::oak::session::SessionConfig;
using ::oak::session::SigningKeyHandle;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

class OakSessionV1ServiceImpl final : public OakSessionV1Service::Service {
 public:
  explicit OakSessionV1ServiceImpl(
      std::unique_ptr<AttestationTokenProvider> provider,
      Server::RequestHandler request_handler)
      : token_provider_(std::move(provider)),
        request_handler_(std::move(request_handler)) {
    CHECK(token_provider_ != nullptr) << "Token provider cannot be null";
    CHECK(request_handler_ != nullptr) << "Request handler cannot be null";
  }

  grpc::Status Stream(grpc::ServerContext* context,
                      grpc::ServerReaderWriter<SessionResponse, SessionRequest>*
                          stream) override {
    LOG(INFO) << "gRPC stream started. Generating key pair via Rust FFI.";

    // 1. Generate ephemeral key pair in Rust.
    std::vector<unsigned char> public_key_bytes(128);
    SigningKeyHandle* private_key_handle = nullptr;

    int public_key_len = generate_key_pair(
        public_key_bytes.data(), public_key_bytes.size(), &private_key_handle);

    if (public_key_len < 0 || private_key_handle == nullptr) {
      LOG(ERROR) << "Failed to generate key pair via Rust FFI (return code "
                 << public_key_len << ").";
      return grpc::Status(grpc::StatusCode::INTERNAL, "Key generation failed");
    }

    public_key_bytes.resize(public_key_len);
    LOG(INFO) << "Generated key pair. Public key size: " << public_key_len;

    // 2. Create nonce from public key and fetch attestation token.
    std::string public_key_str(public_key_bytes.begin(),
                               public_key_bytes.end());
    std::string nonce = absl::Base64Escape(public_key_str);
    LOG(INFO) << "Using public key nonce (Base64): " << nonce;

    absl::StatusOr<std::string> token_or =
        token_provider_->GetAttestationToken(nonce);
    if (!token_or.ok()) {
      LOG(ERROR) << "Failed to get attestation token: " << token_or.status();
      return grpc::Status(grpc::StatusCode::INTERNAL,
                          absl::StrCat("Attestation token fetch failed: ",
                                       token_or.status().ToString()));
    }
    std::string token = *token_or;
    LOG(INFO) << "Successfully fetched attestation token (size "
              << token.length() << ").";

    // 3. Configure Oak Session with the token and private key handle.
    LOG(INFO)
        << "Passing token and key handle to Rust to create SessionConfig.";
    SessionConfig* server_config = create_server_session_config(
        token.c_str(), token.length(), private_key_handle);

    if (server_config == nullptr) {
      LOG(FATAL) << "Rust create_server_session_config returned null.";
      return grpc::Status(grpc::StatusCode::INTERNAL,
                          "Session config creation failed");
    }

    absl::StatusOr<std::unique_ptr<ServerSession>> server_session_or =
        ServerSession::Create(server_config);
    CHECK_OK(server_session_or.status()) << "Failed to create ServerSession";
    std::unique_ptr<ServerSession> server_session =
        std::move(*server_session_or);

    LOG(INFO) << "ServerSession created, entering message processing loop.";

    // 4. Main loop: Read requests, process via Oak, send responses.
    while (true) {
      SessionRequest request;
      if (!stream->Read(&request)) {
        LOG(INFO) << "Client closed the stream.";
        break;
      }
      LOG(INFO) << "gRPC -> Oak: " << request.DebugString();
      absl::Status put_status = server_session->PutIncomingMessage(request);
      CHECK_OK(put_status) << "Failed to process incoming message";

      // Send any handshake or application data generated by Oak.
      CHECK_OK(PumpOutgoingMessages(server_session.get(), stream));

      // Check if we have decrypted application data.
      auto decrypted_message = server_session->ReadToRustBytes();
      if (!decrypted_message.ok()) {
        if (decrypted_message.status().code() != absl::StatusCode::kInternal) {
          LOG(FATAL) << "Failed to read from session: "
                     << decrypted_message.status();
        }
        continue;
      }

      if (!decrypted_message->has_value()) {
        continue;
      }

      std::string decrypted_data =
          static_cast<std::string>(decrypted_message->value());

      // Call the handler to parse and process
      absl::StatusOr<std::string> unencrypted_response_or =
          request_handler_(decrypted_data);
      if (!unencrypted_response_or.ok()) {
        LOG(ERROR) << "Request processing failed: "
                   << unencrypted_response_or.status();
        return grpc::Status(
            grpc::StatusCode::INTERNAL,
            std::string(unencrypted_response_or.status().message()));
      }

      // Send Reply
      LOG(INFO) << "Sending reply (" << unencrypted_response_or->size()
                << " bytes).";
      absl::Status write_status =
          server_session->Write(unencrypted_response_or.value());
      CHECK_OK(write_status) << "Failed to write reply message";

      CHECK_OK(PumpOutgoingMessages(server_session.get(), stream));
    }

    LOG(INFO) << "gRPC stream finished.";
    return grpc::Status::OK;
  }

 private:
  std::unique_ptr<AttestationTokenProvider> token_provider_;
  Server::RequestHandler request_handler_;
};

class ServerImpl : public Server {
 public:
  ServerImpl(std::unique_ptr<OakSessionV1ServiceImpl> service,
             std::unique_ptr<grpc::Server> server, int port)
      : service_(std::move(service)), server_(std::move(server)), port_(port) {}

  virtual ~ServerImpl() override {
    server_->Shutdown();
    server_->Wait();
  }

  virtual void Wait() override { server_->Wait(); }
  virtual int port() override { return port_; }

 private:
  std::unique_ptr<OakSessionV1ServiceImpl> service_;
  std::unique_ptr<grpc::Server> server_;
  int port_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Server>> CreateServer(
    int port,
    std::unique_ptr<AttestationTokenProvider> attestation_token_provider,
    Server::RequestHandler request_handler) {
  if (!attestation_token_provider) {
    return absl::InvalidArgumentError(
        "Attestation tokenb provider cannot be null.");
  }

  if (!request_handler) {
    return absl::InvalidArgumentError("Request handler cannot be null.");
  }

  // Initialize Tink for potential future server-side crypto needs.
  auto status = crypto::tink::TinkConfig::Register();
  CHECK_OK(status) << "Failed to register TinkConfig";

  status = crypto::tink::SignatureConfig::Register();
  CHECK_OK(status) << "Failed to register Tink SignatureConfig";

  std::unique_ptr<OakSessionV1ServiceImpl> service =
      std::make_unique<OakSessionV1ServiceImpl>(
          std::move(attestation_token_provider), std::move(request_handler));

  grpc::ServerBuilder builder;
  std::string server_address = absl::StrFormat("0.0.0.0:%d", port);
  int selected_port;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(service.get());

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on port " << selected_port;

  return std::make_unique<ServerImpl>(std::move(service), std::move(server),
                                      selected_port);
}

}  // namespace confidential_federated_compute::gcp
