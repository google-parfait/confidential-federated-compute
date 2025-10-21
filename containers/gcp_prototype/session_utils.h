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
using ::oak::session::ServerSession;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

// --- Client-Side Utilities ---

/**
 * @brief Reads outgoing messages (Requests) from the ClientSession and writes
 * them to the bidirectional gRPC stream.
 *
 * @param session The active client session state machine.
 * @param stream The bidirectional gRPC stream object.
 * @return true if any message was written, false otherwise.
 */
bool PumpOutgoingMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream);

/**
 * @brief Manages the initial handshake phase for the client session, exchanging
 * messages until the session is established.
 *
 * @param session The active client session.
 * @param stream The bidirectional gRPC stream.
 */
void ExchangeHandshakeMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream);

// --- Server-Side Utilities ---

/**
 * @brief Reads outgoing messages (Responses) from the ServerSession and writes
 * them to the bidirectional gRPC stream.
 *
 * @param session The active server session state machine.
 * @param stream The bidirectional gRPC stream object.
 */
void PumpOutgoingMessages(
    ServerSession* session,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream);

}  // namespace session_utils
}  // namespace gcp_prototype

#endif  // SESSION_UTILS_H
