// session_utils.cc
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

#include "session_utils.h"

namespace gcp_prototype {
namespace session_utils {

// --- Client-Side Utilities ---

bool PumpOutgoingMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream) {
  bool sent_message = false;
  while (true) {
    absl::StatusOr<std::optional<SessionRequest>> request =
        session->GetOutgoingMessage();
    if (!request.ok()) {
      // kInternal means the session needs input before generating output.
      if (request.status().code() != absl::StatusCode::kInternal) {
        LOG(FATAL) << "Client: Failed to get outgoing message: "
                   << request.status();
      }
      break;
    }
    if (!request->has_value()) {
      break;
    }
    LOG(INFO) << "Oak -> gRPC: " << (*request)->DebugString();
    if (!stream->Write(**request)) {
      LOG(ERROR) << "Client: Failed to write message to gRPC stream.";
      return false;
    }
    sent_message = true;
  }
  return sent_message;
}

void ExchangeHandshakeMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream) {
  while (true) {
    // 1. Send any pending outgoing handshake messages (AttestRequest).
    bool sent_message = PumpOutgoingMessages(session, stream);

    // 3. If no message was sent, the handshake is complete (quiesced).
    if (!sent_message) {
      LOG(INFO) << "Handshake quiesced, proceeding to application data.";
      return;
    }

    // 2. Since we sent a message, wait for a reply (HandshakeResponse).
    SessionResponse response;
    if (!stream->Read(&response)) {
      LOG(ERROR) << "Client: Server closed stream during handshake.";
      return;
    }
    LOG(INFO) << "gRPC -> Oak: " << response.DebugString();
    // This call triggers C++ verification via FFI.
    CHECK_OK(session->PutIncomingMessage(response))
        << "Client: Attestation verification or handshake protocol failed";
  }
}

// --- Server-Side Utilities ---

void PumpOutgoingMessages(
    ServerSession* session,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream) {
  while (true) {
    absl::StatusOr<std::optional<SessionResponse>> response =
        session->GetOutgoingMessage();
    if (!response.ok()) {
      // kInternal means the session needs input before generating output.
      if (response.status().code() != absl::StatusCode::kInternal) {
        LOG(FATAL) << "Server: Failed to get outgoing message: "
                   << response.status();
      }
      break;
    }
    if (!response->has_value()) {
      break;
    }

    LOG(INFO) << "Oak -> gRPC: " << (*response)->DebugString();
    if (!stream->Write(**response)) {
      LOG(ERROR) << "Server: Failed to write to gRPC stream (client likely "
                    "disconnected).";
      break;
    }
  }
}

}  // namespace session_utils
}  // namespace gcp_prototype
