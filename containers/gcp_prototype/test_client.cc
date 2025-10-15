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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cc/oak_session/client_session.h"
#include "grpcpp/grpcpp.h"
#include "proto/services/session_v1_service.grpc.pb.h"

using ::oak::services::OakSessionV1Service;
using ::oak::session::AttestationType;
using ::oak::session::ClientSession;
using ::oak::session::HandshakeType;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

const int PORT = 8000;

ABSL_FLAG(std::string, server_address, "localhost",
          "Address of the server to connect to.");

// Helper subroutine to pull messages from the session and send them to the
// server. Returns true if any messages were sent.
bool PumpOutgoingMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream) {
  bool sent_message = false;
  while (true) {
    auto request = session->GetOutgoingMessage();
    if (!request.ok()) {
      if (request.status().code() != absl::StatusCode::kInternal) {
        LOG(FATAL) << "Failed to get outgoing message: " << request.status();
      }
      break;
    }
    if (!request->has_value()) {
      break;
    }
    LOG(INFO) << "Oak -> gRPC: " << (*request)->DebugString();
    if (!stream->Write(**request)) {
      LOG(ERROR) << "Failed to write to stream.";
      return false;  // Treat stream failure as fatal.
    }
    sent_message = true;
  }
  return sent_message;
}

// Main handshake loop.
void ExchangeHandshakeMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream) {
  while (true) {
    // Step 1: Send any available outgoing messages.
    bool sent_message = PumpOutgoingMessages(session, stream);

    // Step 3: If we sent nothing, the handshake has quiesced.
    if (!sent_message) {
      LOG(INFO) << "Handshake quiesced, proceeding to application data.";
      return;
    }

    // Step 2: Since we sent a message, wait for a reply.
    SessionResponse response;
    if (!stream->Read(&response)) {
      LOG(ERROR) << "Server closed stream during handshake.";
      return;
    }
    LOG(INFO) << "gRPC -> Oak: " << response.DebugString();
    CHECK_OK(session->PutIncomingMessage(response));
  }
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  LOG(INFO) << "Initializing Oak ClientSession.";
  SessionConfigBuilder client_builder(AttestationType::kPeerUnidirectional,
                                      HandshakeType::kNoiseNN);
  SessionConfig* client_config = client_builder.Build();
  absl::StatusOr<std::unique_ptr<ClientSession>> client_session_or =
      ClientSession::Create(client_config);
  CHECK_OK(client_session_or.status());
  auto& client_session = *client_session_or;

  std::string server_target =
      absl::StrFormat("%s:%d", absl::GetFlag(FLAGS_server_address), PORT);
  LOG(INFO) << "Connecting to server at " << server_target;

  auto channel =
      grpc::CreateChannel(server_target, grpc::InsecureChannelCredentials());
  auto stub = OakSessionV1Service::NewStub(channel);

  grpc::ClientContext context;
  auto stream = stub->Stream(&context);

  ExchangeHandshakeMessages(client_session.get(), stream.get());

  // Step 3 (from instructions): Handshake is done, write the application
  // message.
  LOG(INFO) << "Client encrypting and sending message.";
  CHECK_OK(client_session->Write("Client says hi!"));

  // Step 4: Do another round of push-pull to send the encrypted message.
  PumpOutgoingMessages(client_session.get(), stream.get());

  // Step 5: Wait for and process the server's reply.
  while (true) {
    // Wait for the server to reply to our application message.
    SessionResponse response;
    if (!stream->Read(&response)) {
      LOG(ERROR) << "Server closed stream while waiting for reply.";
      break;
    }
    LOG(INFO) << "gRPC -> Oak: " << response.DebugString();
    CHECK_OK(client_session->PutIncomingMessage(response));

    // Pull out the decrypted response.
    auto decrypted_message = client_session->ReadToRustBytes();
    CHECK_OK(decrypted_message.status());

    if (decrypted_message->has_value()) {
      LOG(INFO) << "Client decrypted message: "
                << static_cast<std::string>(decrypted_message->value());
      break;  // We got the reply, we're done.
    }
    // If we didn't get a decrypted message, it means we got a session
    // management message (like a post-handshake message). We loop back to
    // process it, which might involve sending a reply.
    PumpOutgoingMessages(client_session.get(), stream.get());
  }

  stream->WritesDone();
  grpc::Status status = stream->Finish();

  if (status.ok()) {
    LOG(INFO) << "RPC finished successfully.";
  } else {
    LOG(ERROR) << "RPC failed: " << status.error_code() << ": "
               << status.error_message();
  }

  return 0;
}
