// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Simple gRPC client that connects to the Oak Noise session server,
// performs attestation verification, and exchanges a single message.

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cc/oak_session/client_session.h"  // Oak ClientSession class
#include "grpcpp/grpcpp.h"
#include "proto/services/session_v1_service.grpc.pb.h"  // Oak gRPC service proto

// FFI header for creating the Oak ClientSession configuration via Rust.
#include "client_session_config.h"
// C++ attestation verifier implementation.
#include "verifier.h"

// Bring relevant namespaces and classes into scope.
using gcp_prototype::AttestationPolicy;
using gcp_prototype::MyVerifier;
using ::oak::services::OakSessionV1Service;
using ::oak::session::ClientSession;
using ::oak::session::SessionConfig;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

// Default server port.
const int PORT = 8000;

// --- Command Line Flags ---

ABSL_FLAG(std::string, server_address, "localhost",
          "Address (IP or hostname) of the server to connect to.");

ABSL_FLAG(std::string, expected_image_digest, "",
          "Expected SHA-256 config digest of the server container image "
          "(e.g., \"sha256:d1df...\"). "
          "If empty, the digest check is skipped by the verifier.");

// --- Helper Functions ---

/**
 * @brief Reads outgoing messages from the Oak session and sends them over the
 * gRPC stream.
 * @param session Pointer to the active Oak ClientSession.
 * @param stream Pointer to the gRPC client stream.
 * @return true if any messages were sent, false if no messages were available
 * or if writing to the stream failed.
 */
bool PumpOutgoingMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream) {
  bool sent_message = false;
  while (true) {
    // Attempt to get the next message from the session state machine.
    absl::StatusOr<std::optional<SessionRequest>> request =
        session->GetOutgoingMessage();
    if (!request.ok()) {
      // kInternal usually means the session needs more input (e.g., from the
      // server) before it can produce more output. Other errors are fatal.
      if (request.status().code() != absl::StatusCode::kInternal) {
        LOG(FATAL) << "Failed to get outgoing message: " << request.status();
      }
      break;  // No message available right now.
    }
    if (!request->has_value()) {
      break;  // No message available right now.
    }

    // Log and send the message.
    LOG(INFO) << "Oak -> gRPC: " << (*request)->DebugString();
    if (!stream->Write(**request)) {
      LOG(ERROR) << "Failed to write message to gRPC stream.";
      return false;  // Treat stream writing failure as fatal for this pump.
    }
    sent_message = true;
  }
  return sent_message;  // Indicate if we actually sent anything.
}

/**
 * @brief Manages the initial handshake phase (including attestation) of the
 * Oak Noise session.
 *
 * This function loops, sending messages from the client session to the server
 * and feeding received server messages back into the client session, until the
 * handshake quiesces (no more messages need to be exchanged for the handshake).
 * Attestation verification failures within the session's PutIncomingMessage
 * call will cause a fatal CHECK failure.
 *
 * @param session Pointer to the active Oak ClientSession.
 * @param stream Pointer to the gRPC client stream.
 */
void ExchangeHandshakeMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream) {
  while (true) {
    // Step 1: Send any pending outgoing handshake messages (e.g.,
    // attest_request).
    bool sent_message = PumpOutgoingMessages(session, stream);

    // Step 3: If we didn't send anything, the handshake requires input from the
    // server or is complete (quiesced).
    if (!sent_message) {
      LOG(INFO) << "Handshake quiesced, proceeding to application data.";
      return;  // Exit the handshake loop.
    }

    // Step 2: Since we sent a message, wait for a reply from the server.
    SessionResponse response;
    if (!stream->Read(&response)) {
      // If the stream closes during handshake, it's an error.
      LOG(ERROR) << "Server closed stream during handshake.";
      // Future work (b/452094015): Return an error status instead of just
      // logging.
      return;
    }
    LOG(INFO) << "gRPC -> Oak: " << response.DebugString();

    // Feed the server's response into the client session state machine.
    // This is where attestation verification happens internally based on the
    // verifier configured via Rust FFI. If verification fails, this CHECK
    // fails.
    CHECK_OK(session->PutIncomingMessage(response))
        << "Attestation verification or handshake protocol failed";
  }
}

// --- Main ---

int main(int argc, char** argv) {
  // Parse command line flags (server_address, expected_image_digest).
  absl::ParseCommandLine(argc, argv);

  // --- 1. Configure Attestation Policy ---
  AttestationPolicy policy;
  // Set the expected image digest based on the command-line flag.
  policy.expected_image_digest = absl::GetFlag(FLAGS_expected_image_digest);

  // --- THIS IS THE FIX ---
  // Explicitly override the new default. For this test client, we ALLOW
  // the server to be in debug mode. A real client would omit this line.
  policy.require_debug_disabled = false;
  // -----------------------

  // --- 2. Initialize Verifier ---
  MyVerifier verifier;
  verifier.SetPolicy(policy);  // Apply the configured policy.
  // Fetch Google's public keys (JWKS). CHECK_OK ensures failure is fatal.
  CHECK_OK(verifier.Initialize())
      << "Failed to initialize attestation verifier";

  // --- 3. Create Oak ClientSession Config via Rust FFI ---
  LOG(INFO) << "Initializing Oak ClientSession via Rust.";
  // Call the Rust function, passing the C++ MyVerifier instance as opaque
  // context and the verify_jwt_f function pointer as the callback. Rust will
  // call verify_jwt_f during the handshake, which will invoke
  // MyVerifier::VerifyJwt.
  SessionConfig* client_config = create_client_session_config(
      static_cast<void*>(&verifier),  // Pass verifier object as context.
      &verify_jwt_f                   // Pass C++ verification function pointer.
  );

  // --- 4. Create Oak Client Session ---
  // ClientSession::Create takes ownership of the client_config pointer.
  absl::StatusOr<std::unique_ptr<ClientSession>> client_session_or =
      ClientSession::Create(client_config);
  CHECK_OK(client_session_or.status()) << "Failed to create ClientSession";
  // Use a reference for convenience.
  ClientSession& client_session = **client_session_or;

  // --- 5. Connect to Server ---
  std::string server_target =
      absl::StrFormat("%s:%d", absl::GetFlag(FLAGS_server_address), PORT);
  LOG(INFO) << "Connecting to server at " << server_target;
  // Create an insecure gRPC channel (no TLS at the gRPC layer, encryption is
  // handled by the Oak Noise session).
  auto channel =
      grpc::CreateChannel(server_target, grpc::InsecureChannelCredentials());
  // Create a stub for the Oak Session service.
  auto stub = OakSessionV1Service::NewStub(channel);
  grpc::ClientContext context;
  // Start the bidirectional gRPC stream.
  auto stream = stub->Stream(&context);

  // --- 6. Perform Handshake & Attestation ---
  ExchangeHandshakeMessages(&client_session, stream.get());
  // If this function returns, the handshake was successful and attestation
  // passed.

  // --- 7. Send Application Data ---
  LOG(INFO) << "Client encrypting and sending application message.";
  CHECK_OK(client_session.Write("Client says hi!"));
  // Ensure the encrypted message is sent over the stream.
  PumpOutgoingMessages(&client_session, stream.get());

  // --- 8. Receive Application Data ---
  LOG(INFO) << "Waiting for server reply...";
  while (true) {
    SessionResponse response;
    // Wait for a message from the server.
    if (!stream->Read(&response)) {
      LOG(ERROR) << "Server closed stream while waiting for application reply.";
      break;  // Exit loop on stream closure.
    }
    LOG(INFO) << "gRPC -> Oak: " << response.DebugString();
    // Feed the message to the session (could be post-handshake management).
    CHECK_OK(client_session.PutIncomingMessage(response));

    // --- FIX: Reverted to using ReadToRustBytes() ---
    // This matches the baseline code and correctly uses the CXX bridge.
    // It returns a StatusOr<optional<rust::Vec<uint8_t>>>
    auto decrypted_message = client_session.ReadToRustBytes();
    CHECK_OK(decrypted_message.status()) << "Failed to read from session";

    if (decrypted_message->has_value()) {
      // Successfully decrypted the application message.
      // Use static_cast<std::string> to convert the rust::Vec<uint8_t>.
      LOG(INFO) << "Client decrypted message: "
                << static_cast<std::string>(decrypted_message->value());
      // --- END FIX ---
      break;  // We got the reply, exit the loop.
    }
    // If Read() returned an empty optional, it might have processed a
    // non-application message. Pump outgoing messages just in case a response
    // is needed, then loop back to read the next message from the server.
    PumpOutgoingMessages(&client_session, stream.get());
  }

  // --- 9. Clean Shutdown ---
  LOG(INFO) << "Closing stream and exiting.";
  stream->WritesDone();                    // Signal client is done writing.
  grpc::Status status = stream->Finish();  // Wait for server to finish.

  if (status.ok()) {
    LOG(INFO) << "RPC finished successfully.";
  } else {
    // Log gRPC-level errors.
    LOG(ERROR) << "RPC failed: " << status.error_code() << ": "
               << status.error_message();
    return 1;  // Indicate failure.
  }

  return 0;  // Indicate success.
}
