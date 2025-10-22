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
#include "absl/strings/escaping.h"  // For Base64Escape
#include "absl/strings/str_format.h"
#include "attestation.h"        // Helper for getting attestation token.
#include "cc/ffi/rust_bytes.h"  // For rust::Vec and its C++ bindings.
#include "cc/oak_session/server_session.h"  // Oak ServerSession class.
#include "grpcpp/grpcpp.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "proto/services/session_v1_service.grpc.pb.h"  // Oak gRPC service proto.
#include "server_session_config.h"  // FFI header for Rust config/key generation.
#include "tink/config/tink_config.h"          // For initializing Tink.
#include "tink/signature/signature_config.h"  // For initializing Tink Signatures.

// Using declarations for convenience.
using ::oak::services::OakSessionV1Service;
using ::oak::session::ServerSession;
using ::oak::session::SessionConfig;
using ::oak::session::SigningKeyHandle;  // Opaque handle type from FFI header.
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

// Default server port.
const int PORT = 8000;

// Command-line flag for testing without a real attestation agent.
ABSL_FLAG(bool, use_fake_attestation_token, false,
          "If true, uses a hardcoded fake attestation token instead of "
          "fetching one from the Confidential Space agent.");

/**
 * @brief Reads outgoing messages from the Oak session and writes them to the
 * gRPC stream.
 *
 * Continues pumping messages until the session has no more messages ready to
 * send (indicated by a kInternal status or empty optional).
 *
 * @param session Pointer to the active Oak ServerSession.
 * @param stream Pointer to the gRPC server stream.
 */
void PumpOutgoingMessages(
    ServerSession* session,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream) {
  while (true) {
    // Attempt to get the next message from the session state machine.
    absl::StatusOr<std::optional<SessionResponse>> response =
        session->GetOutgoingMessage();
    if (!response.ok()) {
      // kInternal usually means the session needs more input. Other errors are
      // fatal.
      if (response.status().code() != absl::StatusCode::kInternal) {
        LOG(FATAL) << "Failed to get outgoing message: " << response.status();
      }
      break;  // No message available right now.
    }
    if (!response->has_value()) {
      break;  // No message available right now.
    }

    // Log and send the message.
    LOG(INFO) << "Oak -> gRPC: " << (*response)->DebugString();
    if (!stream->Write(**response)) {
      // If writing to the stream fails, the client likely disconnected.
      LOG(ERROR)
          << "Failed to write to gRPC stream (client likely disconnected).";
      break;  // Stop pumping for this stream.
    }
  }
}

/**
 * @brief Implementation of the Oak Session gRPC service.
 * Handles a single client stream connection.
 */
class OakSessionV1ServiceImpl final : public OakSessionV1Service::Service {
 public:
  /**
   * @brief Handles the bidirectional gRPC stream for an Oak Noise session.
   *
   * This function performs the following steps for each client connection:
   * 1. Generates a P-256 key pair using Rust FFI.
   * 2. Fetches a GCP Confidential Space attestation token (JWT), using the
   * public key as the nonce.
   * 3. Creates the Oak ServerSession configuration using Rust FFI, providing
   * the token and the private key handle.
   * 4. Creates the Oak ServerSession object.
   * 5. Enters a loop to process messages:
   * - Reads incoming messages from the client stream.
   * - Feeds incoming messages into the Oak session state machine.
   * - Pumps any resulting outgoing messages back to the client.
   * - Attempts to read decrypted application data from the session.
   * - If data is received, processes it (logs it) and sends a reply.
   *
   * @param context The gRPC server context.
   * @param stream The bidirectional stream for session messages.
   * @return grpc::Status::OK on successful stream completion, or an error
   * status on failure.
   */
  grpc::Status Stream(grpc::ServerContext* context,
                      grpc::ServerReaderWriter<SessionResponse, SessionRequest>*
                          stream) override {
    LOG(INFO) << "gRPC stream started. Generating key pair via Rust FFI.";

    // --- 1. Generate Key Pair ---
    std::vector<unsigned char> public_key_bytes(
        128);  // Ample buffer for P-256 key.
    SigningKeyHandle* private_key_handle = nullptr;  // Opaque handle from Rust.
    // Call Rust FFI to generate keys.
    int public_key_len = generate_key_pair(
        public_key_bytes.data(), public_key_bytes.size(), &private_key_handle);
    // Check for errors during key generation.
    if (public_key_len < 0 || private_key_handle == nullptr) {
      LOG(ERROR) << "Failed to generate key pair via Rust FFI (return code "
                 << public_key_len << ").";
      // Return an internal error to the client.
      return grpc::Status(grpc::StatusCode::INTERNAL, "Key generation failed");
    }
    // Resize vector to actual key length (should be 65 for P-256 uncompressed).
    public_key_bytes.resize(public_key_len);
    LOG(INFO) << "Generated key pair. Public key size: " << public_key_len;

    // --- 2. Prepare Attestation Token ---
    std::string token;
    if (absl::GetFlag(FLAGS_use_fake_attestation_token)) {
      // Use a fake token if the flag is set (for local testing).
      LOG(WARNING) << "Using fake attestation token for testing.";
      token = "FAKE_JWT_TOKEN_FROM_C++_SERVER";
    } else {
      // Use the real attestation token flow.
      // The nonce *must* be the Base64 encoding of the raw public key bytes.
      std::string public_key_str(public_key_bytes.begin(),
                                 public_key_bytes.end());
      std::string nonce = absl::Base64Escape(public_key_str);
      LOG(INFO) << "Using public key nonce (Base64): " << nonce;

      // Call the attestation helper function to fetch the token from the agent.
      absl::StatusOr<std::string> token_or =
          gcp_prototype::GetAttestationToken(nonce);
      if (!token_or.ok()) {
        LOG(FATAL) << "Failed to get attestation token: " << token_or.status();
        // Return an internal error to the client.
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "Attestation token fetch failed");
      }
      token = *token_or;
      LOG(INFO) << "Successfully fetched attestation token (size "
                << token.length() << ").";
    }

    // --- 3. Create Oak Server Session Config ---
    LOG(INFO)
        << "Passing token and key handle to Rust to create SessionConfig.";
    // Call Rust FFI, passing the token and the private key handle.
    // Ownership of the private key handle is transferred to Rust here.
    SessionConfig* server_config = create_server_session_config(
        token.c_str(), token.length(), private_key_handle);
    if (server_config == nullptr) {
      // Rust function should ideally not return null on success, but check
      // defensively.
      LOG(FATAL) << "Rust create_server_session_config returned null.";
      return grpc::Status(grpc::StatusCode::INTERNAL,
                          "Session config creation failed");
    }

    // --- 4. Create Oak Server Session ---
    // ServerSession::Create takes ownership of the server_config pointer.
    absl::StatusOr<std::unique_ptr<ServerSession>> server_session_or =
        ServerSession::Create(server_config);
    // Use CHECK_OK for fatal errors during setup.
    CHECK_OK(server_session_or.status()) << "Failed to create ServerSession";
    std::unique_ptr<ServerSession> server_session =
        std::move(*server_session_or);

    // --- 5. Main Message Processing Loop ---
    LOG(INFO) << "ServerSession created, entering message processing loop.";
    while (true) {
      // Step A: Read a message from the client stream.
      SessionRequest request;
      if (!stream->Read(&request)) {
        // Read failed, usually means the client closed the stream gracefully.
        LOG(INFO) << "Client closed the stream.";
        break;  // Exit the loop.
      }
      LOG(INFO) << "gRPC -> Oak: " << request.DebugString();

      // Step B: Feed the incoming message into the Oak session.
      absl::Status put_status = server_session->PutIncomingMessage(request);
      // CHECK_OK here handles potential decryption/protocol errors during
      // handshake.
      CHECK_OK(put_status) << "Failed to process incoming message";

      // Step C: Pump any resulting outgoing messages (e.g., handshake
      // responses) back to the client.
      PumpOutgoingMessages(server_session.get(), stream);

      // Step D: Attempt to read decrypted application data using
      // ReadToRustBytes. This uses the CXX bridge to avoid copying data from
      // Rust to C++.
      auto decrypted_message = server_session->ReadToRustBytes();
      if (!decrypted_message.ok()) {
        // kInternal means the session is not ready to read yet (e.g., handshake
        // not complete). Other errors are fatal.
        if (decrypted_message.status().code() != absl::StatusCode::kInternal) {
          LOG(FATAL) << "Failed to read from session: "
                     << decrypted_message.status();
        }
        // If kInternal, just continue the loop to wait for more messages.
        continue;
      }

      if (!decrypted_message->has_value()) {
        // No application message available yet, loop back to wait for more
        // input from the client or allow session management tasks.
        continue;
      }

      // Step E: Application message received! Process and reply.
      // Use static_cast<std::string> to leverage the conversion operator
      // provided by CXX for rust::Vec<uint8_t>.
      LOG(INFO) << "Server decrypted application message: \""
                << static_cast<std::string>(decrypted_message->value()) << "\"";

      // Simple echo/reply logic.
      LOG(INFO) << "Server encrypting and sending application reply.";
      absl::Status write_status = server_session->Write("Server says hi back!");
      CHECK_OK(write_status) << "Failed to write reply message";

      // Step F: Pump the encrypted reply back to the client.
      PumpOutgoingMessages(server_session.get(), stream);
    }  // End of message processing loop.

    LOG(INFO) << "gRPC stream finished.";
    return grpc::Status::OK;  // Indicate graceful stream termination.
  }
};

/**
 * @brief Sets up and runs the gRPC server.
 */
void RunServer() {
  std::string server_address = absl::StrFormat("0.0.0.0:%d", PORT);
  OakSessionV1ServiceImpl service;  // The service implementation.

  grpc::ServerBuilder builder;
  // Listen on the specified address without transport security (encryption is
  // handled by the Oak Noise session layer).
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Build and start the server.
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;

  // Wait for the server to shut down (blocking call).
  server->Wait();
}

int main(int argc, char** argv) {
  // Initialize logging and parse command-line flags.
  absl::ParseCommandLine(argc, argv);

  // Initialize Tink primitives required by the application.
  // Although key generation/signing happens in Rust, Tink might be used
  // elsewhere (e.g., if client attestation were added), so standard
  // initialization is good practice. CHECK_OK ensures failure is fatal.
  auto status = crypto::tink::TinkConfig::Register();
  CHECK_OK(status) << "Failed to register TinkConfig";
  status = crypto::tink::SignatureConfig::Register();
  CHECK_OK(status) << "Failed to register Tink SignatureConfig";

  // Run the server loop.
  RunServer();

  return 0;
}
