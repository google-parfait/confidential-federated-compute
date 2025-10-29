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

void ExchangeHandshakeMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream) {
  while (true) {
    // 1. Send any pending outgoing handshake messages.
    absl::StatusOr<bool> sent_or = PumpOutgoingMessages(session, stream);
    CHECK_OK(
        sent_or);  // Crash on fatal session/network errors during handshake.

    // 2. If no message was sent, the handshake is complete (quiesced).
    if (!*sent_or) {
      LOG(INFO) << "Handshake quiesced, proceeding to application data.";
      return;
    }

    // 3. Since we sent a message, expect a reply.
    SessionResponse response;
    if (!stream->Read(&response)) {
      LOG(FATAL) << "Client: Server closed stream during handshake.";
    }
    LOG(INFO) << "gRPC -> Oak: " << response.DebugString();
    CHECK_OK(session->PutIncomingMessage(response))
        << "Client: Attestation verification or handshake protocol failed";
  }
}

}  // namespace session_utils
}  // namespace gcp_prototype
