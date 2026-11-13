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

#include <fstream>
#include <iostream>
#include <optional>

namespace confidential_federated_compute::gcp {

absl::Status ExchangeHandshakeMessages(
    ClientSession* session,
    grpc::ClientReaderWriter<SessionRequest, SessionResponse>* stream) {
  // Strict request/response exchange: exactly one GetOutgoingMessage per
  // round. Do not use PumpOutgoingMessages here: re-polling during the
  // attestation step makes Oak append the self-assertions to the attestation
  // binding token a second time, which breaks session binding verification.
  int round = 0;
  while (!session->IsOpen()) {
    round++;
    std::cerr << "----------> ExchangeHandshake round " << round
              << ": calling GetOutgoingMessage" << std::endl;

    // 1. Get exactly one outgoing handshake message.
    absl::StatusOr<std::optional<SessionRequest>> request_or =
        session->GetOutgoingMessage();
    if (!request_or.ok()) {
      std::cerr << "----------> ExchangeHandshake round " << round
                << ": GetOutgoingMessage FAILED: "
                << request_or.status().ToString() << std::endl;
      return absl::InternalError(absl::StrCat("Error during handshake: ",
                                              request_or.status().ToString()));
    }
    if (!request_or->has_value()) {
      std::cerr << "----------> ExchangeHandshake round " << round
                << ": no outgoing message but session is not open" << std::endl;
      return absl::InternalError(
          "Handshake stalled: session produced no outgoing message but is "
          "not open.");
    }

    if (!stream->Write(**request_or)) {
      std::cerr << "----------> ExchangeHandshake round " << round
                << ": stream->Write failed" << std::endl;
      return absl::UnavailableError(
          "Failed to write handshake message to gRPC stream.");
    }

    std::cerr << "----------> ExchangeHandshake round " << round
              << ": sent 1 message, waiting for server response (stream->Read)"
              << std::endl;

    // Diagnostic: snapshot memory before Read in case of OOM kill.
    {
      std::ifstream proc_status("/proc/self/status");
      std::string line;
      while (std::getline(proc_status, line)) {
        if (line.find("VmRSS") != std::string::npos ||
            line.find("VmSize") != std::string::npos ||
            line.find("VmPeak") != std::string::npos) {
          std::cerr << "----------> PRE-READ " << line << std::endl;
        }
      }
    }

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

  std::cerr << "----------> ExchangeHandshake: session open after " << round
            << " rounds, proceeding to application data." << std::endl;
  LOG(INFO) << "Handshake complete, proceeding to application data.";
  return absl::OkStatus();
}

}  // namespace confidential_federated_compute::gcp
