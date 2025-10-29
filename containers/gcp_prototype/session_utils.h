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

#ifndef SESSION_UTILS_H
#define SESSION_UTILS_H

#include <grpcpp/grpcpp.h>

#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "cc/oak_session/client_session.h"
#include "cc/oak_session/server_session.h"
#include "proto/services/session_v1_service.grpc.pb.h"

namespace gcp_prototype {
namespace session_utils {

using ::oak::session::ClientSession;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

/**
 * @brief Reads outgoing messages from any Oak Session and writes them to the
 * provided gRPC stream.
 *
 * @tparam SessionT The Oak session type (ClientSession or ServerSession).
 * @tparam StreamT The gRPC stream type.
 * @param session The active session state machine.
 * @param stream The gRPC stream to write to.
 * @return absl::StatusOr<bool> True if at least one message was sent, false if
 * none were sent, or an error status.
 */
template <typename SessionT, typename StreamT>
absl::StatusOr<bool> PumpOutgoingMessages(SessionT* session, StreamT* stream) {
  bool sent_any_message = false;
  while (true) {
    auto outgoing_message = session->GetOutgoingMessage();
    if (!outgoing_message.ok()) {
      // kInternal means the session needs more input before it can generate
      // output. This is a normal state, not an error.
      if (outgoing_message.status().code() == absl::StatusCode::kInternal) {
        return sent_any_message;
      }
      return outgoing_message.status();
    }
    if (!outgoing_message->has_value()) {
      return sent_any_message;
    }

    LOG(INFO) << "Oak -> gRPC: " << (*outgoing_message)->DebugString();
    if (!stream->Write(**outgoing_message)) {
      return absl::UnavailableError("Failed to write to gRPC stream.");
    }
    sent_any_message = true;
  }
}

/**
 * @brief Manages the initial handshake phase for the client session.
 */
void ExchangeHandshakeMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream);

}  // namespace session_utils
}  // namespace gcp_prototype

#endif  // SESSION_UTILS_H
