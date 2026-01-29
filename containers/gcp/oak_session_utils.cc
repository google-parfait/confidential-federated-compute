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

#include "oak_session_utils.h"

namespace confidential_federated_compute::gcp {

absl::Status ExchangeHandshakeMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream) {
  while (true) {
    // 1. Send any pending outgoing handshake messages.
    absl::StatusOr<bool> sent_or = PumpOutgoingMessages(session, stream);
    if (!sent_or.ok()) {
      return absl::InternalError(absl::StrCat("Error during handshake: ",
                                              sent_or.status().ToString()));
    }

    if (!*sent_or) {
      LOG(INFO) << "Handshake quiesced, proceeding to application data.";
      return absl::OkStatus();
    }

    SessionResponse response;
    if (!stream->Read(&response)) {
      return absl::InternalError("Stream was closed during handshake.");
    }

    absl::Status put_status = session->PutIncomingMessage(response);
    if (!put_status.ok()) {
      return absl::InternalError(absl::StrCat(
          "Attestation verification or handshake protocol failed: ",
          put_status.ToString()));
    }
  }
}

}  // namespace confidential_federated_compute::gcp
