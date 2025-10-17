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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cc/oak_session/server_session.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "proto/services/session_v1_service.grpc.pb.h"

using ::oak::services::OakSessionV1Service;
using ::oak::session::AttestationType;
using ::oak::session::HandshakeType;
using ::oak::session::ServerSession;
using ::oak::session::SessionConfig;
using ::oak::session::SessionConfigBuilder;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

const int PORT = 8000;

// Helper function to pull all available messages from the session and send
// them.
void PumpOutgoingMessages(
    ServerSession* session,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream) {
  while (true) {
    auto response = session->GetOutgoingMessage();
    if (!response.ok()) {
      if (response.status().code() != absl::StatusCode::kInternal) {
        LOG(FATAL) << "Failed to get outgoing message: " << response.status();
      }
      // kInternal means the session is waiting for more input, so we stop.
      break;
    }
    if (!response->has_value()) {
      // No more messages to send for now.
      break;
    }
    LOG(INFO) << "Oak -> gRPC: " << (*response)->DebugString();
    if (!stream->Write(**response)) {
      LOG(ERROR) << "Failed to write to stream.";
      break;
    }
  }
}

class OakSessionV1ServiceImpl final : public OakSessionV1Service::Service {
 public:
  grpc::Status Stream(grpc::ServerContext* context,
                      grpc::ServerReaderWriter<SessionResponse, SessionRequest>*
                          stream) override {
    LOG(INFO) << "gRPC stream started. Initializing Oak ServerSession.";

    SessionConfigBuilder server_builder(AttestationType::kSelfUnidirectional,
                                        HandshakeType::kNoiseNN);
    SessionConfig* server_config = server_builder.Build();
    absl::StatusOr<std::unique_ptr<ServerSession>> server_session_or =
        ServerSession::Create(server_config);
    CHECK_OK(server_session_or.status());
    auto& server_session = *server_session_or;

    // The main server loop.
    while (true) {
      // Step 1: Wait for a message from the client.
      SessionRequest request;
      if (!stream->Read(&request)) {
        LOG(INFO) << "Client closed the stream.";
        break;
      }
      LOG(INFO) << "gRPC -> Oak: " << request.DebugString();
      CHECK_OK(server_session->PutIncomingMessage(request));

      // Step 2: Send back any resulting messages.
      PumpOutgoingMessages(server_session.get(), stream);

      // Step 3 & 4: Try to read a decrypted message. If we can't, loop back.
      auto decrypted_message = server_session->ReadToRustBytes();
      if (!decrypted_message.ok()) {
        if (decrypted_message.status().code() != absl::StatusCode::kInternal) {
          LOG(FATAL) << "Failed to read from session: "
                     << decrypted_message.status();
        }
        // kInternal means the session is not open yet, so we continue.
        continue;
      }

      if (!decrypted_message->has_value()) {
        // No application message available yet, loop back to wait for more
        // from the client.
        continue;
      }

      // Step 5: We got a message! Process it and send a reply.
      LOG(INFO) << "Server decrypted message: "
                << static_cast<std::string>(decrypted_message->value());

      LOG(INFO) << "Server encrypting and sending reply.";
      CHECK_OK(server_session->Write("Server says hi back!"));

      // Loop back to Step 2 to ensure the encrypted reply is sent.
      PumpOutgoingMessages(server_session.get(), stream);
    }

    LOG(INFO) << "gRPC stream finished.";
    return grpc::Status::OK;
  }
};

void RunServer() {
  std::string server_address = absl::StrFormat("0.0.0.0:%d", PORT);
  OakSessionV1ServiceImpl service;

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;
  server->Wait();
}

int main(int argc, char** argv) {
  RunServer();
  return 0;
}
