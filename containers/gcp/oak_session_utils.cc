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

#include <iostream>

namespace confidential_federated_compute::gcp {

absl::Status ExchangeHandshakeMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream) {
  int round = 0;
  while (true) {
    round++;
    std::cerr << "----------> ExchangeHandshake round " << round
              << ": calling PumpOutgoingMessages" << std::endl;

    // 1. Send any pending outgoing handshake messages.
    absl::StatusOr<bool> sent_or = PumpOutgoingMessages(session, stream);
    if (!sent_or.ok()) {
      std::cerr << "----------> ExchangeHandshake round " << round
                << ": PumpOutgoingMessages FAILED: "
                << sent_or.status().ToString() << std::endl;
      return absl::InternalError(absl::StrCat("Error during handshake: ",
                                              sent_or.status().ToString()));
    }

    std::cerr << "----------> ExchangeHandshake round " << round
              << ": PumpOutgoingMessages returned sent=" << *sent_or
              << std::endl;

    if (!*sent_or) {
      std::cerr << "----------> ExchangeHandshake: handshake quiesced after "
                << round << " rounds, proceeding to application data."
                << std::endl;
      LOG(INFO) << "Handshake quiesced, proceeding to application data.";
      return absl::OkStatus();
    }

    std::cerr << "----------> ExchangeHandshake round " << round
              << ": waiting for server response (stream->Read)" << std::endl;

    SessionResponse response;
    if (!stream->Read(&response)) {
      std::cerr << "----------> ExchangeHandshake round " << round
                << ": stream->Read returned false, stream closed!" << std::endl;
      return absl::InternalError("Stream was closed during handshake.");
    }

    std::cerr << "----------> ExchangeHandshake round " << round
              << ": got server response, size=" << response.ByteSizeLong()
              << " bytes" << std::endl;

    std::cerr << "----------> ExchangeHandshake round " << round
              << ": calling session->PutIncomingMessage" << std::endl;

    absl::Status put_status = session->PutIncomingMessage(response);

    std::cerr << "----------> ExchangeHandshake round " << round
              << ": PutIncomingMessage returned: " << put_status.ToString()
              << std::endl;

    if (!put_status.ok()) {
      std::cerr << "----------> ExchangeHandshake round " << round
                << ": ATTESTATION/HANDSHAKE FAILED: " << put_status.ToString()
                << std::endl;
      return absl::InternalError(absl::StrCat(
          "Attestation verification or handshake protocol failed: ",
          put_status.ToString()));
    }
  }
}

}  // namespace confidential_federated_compute::gcp
