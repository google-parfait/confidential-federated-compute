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
#include "absl/time/time.h"
#include "cc/oak_session/client_session.h"
#include "client_session_config.h"
#include "grpcpp/grpcpp.h"
#include "proto/services/session_v1_service.grpc.pb.h"
#include "session_utils.h"
#include "verifier.h"

using gcp_prototype::AttestationPolicy;
using gcp_prototype::MyVerifier;
using ::oak::services::OakSessionV1Service;
using ::oak::session::ClientSession;
using ::oak::session::SessionConfig;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

const int PORT = 8000;

ABSL_FLAG(std::string, server_address, "localhost", "Server address.");
ABSL_FLAG(std::string, expected_image_digest, "", "Expected image digest.");
ABSL_FLAG(bool, skip_policy_enforcement, false, "Skip policy checks (DEBUG).");
ABSL_FLAG(bool, dump_jwt, false, "Dump JWT payload (DEBUG).");
ABSL_FLAG(std::string, verifier_type, "ita", "Verifier type: 'ita' or 'gca'.");
ABSL_FLAG(std::string, expected_project_id, "",
          "Expected GCP Project ID the workload must run in.");
ABSL_FLAG(std::string, expected_service_account, "",
          "Expected GCP Service Account the workload must use.");
ABSL_FLAG(bool, require_hw_tcb_uptodate, true,
          "Require Intel HW TCB to be UpToDate. Set to false for preview/test "
          "environments.");
ABSL_FLAG(std::string, min_sw_tcb_date, "",
          "Minimum acceptable SW TCB date (RFC3339 format, e.g., "
          "2024-01-01T00:00:00Z).");
ABSL_FLAG(std::string, min_hw_tcb_date, "",
          "Minimum acceptable HW TCB date (RFC3339 format, e.g., "
          "2024-01-01T00:00:00Z).");

// Parses an optional date flag. Fails immediately if the format is invalid to
// prevent runtime errors later during policy enforcement.
std::optional<absl::Time> ParseOptionalDateFlag(absl::string_view flag_name,
                                                const std::string& date_str) {
  if (date_str.empty()) {
    return std::nullopt;
  }
  absl::Time parsed_time;
  std::string err;
  // Use RFC3339_full to strictly enforce ISO 8601 compliance.
  if (!absl::ParseTime(absl::RFC3339_full, date_str, &parsed_time, &err)) {
    LOG(FATAL) << "Invalid format for --" << flag_name << " ('" << date_str
               << "'): " << err
               << ". Expected RFC3339 format (e.g., 2024-01-01T00:00:00Z).";
  }
  return parsed_time;
}

// Builds the attestation policy from command-line flags.
AttestationPolicy BuildPolicyFromFlags() {
  AttestationPolicy policy;
  std::string verifier_type_flag = absl::GetFlag(FLAGS_verifier_type);
  if (verifier_type_flag == "ita") {
    policy.verifier_type = AttestationPolicy::VerifierType::kIta;
  } else if (verifier_type_flag == "gca") {
    policy.verifier_type = AttestationPolicy::VerifierType::kGca;
  } else {
    LOG(FATAL) << "Invalid --verifier_type: must be 'ita' or 'gca'.";
  }

  policy.expected_image_digest = absl::GetFlag(FLAGS_expected_image_digest);
  policy.expected_project_id = absl::GetFlag(FLAGS_expected_project_id);
  policy.expected_service_account =
      absl::GetFlag(FLAGS_expected_service_account);
  policy.require_hw_tcb_uptodate = absl::GetFlag(FLAGS_require_hw_tcb_uptodate);

  policy.min_sw_tcb_date = ParseOptionalDateFlag(
      "min_sw_tcb_date", absl::GetFlag(FLAGS_min_sw_tcb_date));
  policy.min_hw_tcb_date = ParseOptionalDateFlag(
      "min_hw_tcb_date", absl::GetFlag(FLAGS_min_hw_tcb_date));

  // Override policy default to allow debug mode for this test client.
  // In a production environment, this should likely be true by default.
  policy.require_debug_disabled = false;

  return policy;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  using gcp_prototype::session_utils::ExchangeHandshakeMessages;
  using gcp_prototype::session_utils::PumpOutgoingMessages;

  // 1. Configure & Initialize Verifier
  MyVerifier verifier;
  verifier.SetPolicy(BuildPolicyFromFlags());
  verifier.SkipPolicyEnforcement(absl::GetFlag(FLAGS_skip_policy_enforcement));
  verifier.SetDumpJwt(absl::GetFlag(FLAGS_dump_jwt));

  // Fetch JWKS and prepare Tink for verification.
  CHECK_OK(verifier.Initialize())
      << "Failed to initialize attestation verifier";

  // 2. Initialize Session & Connect
  LOG(INFO) << "Initializing Oak ClientSession via Rust.";
  // We use FFI here because the Oak ClientSession is implemented in Rust.
  // We pass the C++ verifier object and a callback function so that Rust can
  // delegate the actual JWT verification back to our C++ implementation.
  SessionConfig* client_config = create_client_session_config(
      static_cast<void*>(&verifier), &verify_jwt_f);

  absl::StatusOr<std::unique_ptr<ClientSession>> client_session_or =
      ClientSession::Create(client_config);
  CHECK_OK(client_session_or.status()) << "Failed to create ClientSession";
  ClientSession& client_session = **client_session_or;

  std::string server_target =
      absl::StrFormat("%s:%d", absl::GetFlag(FLAGS_server_address), PORT);
  LOG(INFO) << "Connecting to server at " << server_target;

  auto channel =
      grpc::CreateChannel(server_target, grpc::InsecureChannelCredentials());
  auto stub = OakSessionV1Service::NewStub(channel);
  grpc::ClientContext context;
  auto stream = stub->Stream(&context);

  // 3. Execute Flow: Handshake, send message, receive response.
  // The handshake will trigger the Rust session to call our C++ verification
  // callback.
  ExchangeHandshakeMessages(&client_session, stream.get());

  LOG(INFO) << "Client encrypting and sending application message.";
  CHECK_OK(client_session.Write("Client says hi!"));
  PumpOutgoingMessages(&client_session, stream.get());

  LOG(INFO) << "Waiting for server reply...";
  while (true) {
    SessionResponse response;
    if (!stream->Read(&response)) {
      LOG(ERROR) << "Server closed stream while waiting for application reply.";
      break;
    }
    LOG(INFO) << "gRPC -> Oak: " << response.DebugString();
    CHECK_OK(client_session.PutIncomingMessage(response));
    auto decrypted_message = client_session.ReadToRustBytes();
    CHECK_OK(decrypted_message.status()) << "Failed to read from session";

    if (decrypted_message->has_value()) {
      LOG(INFO) << "Client decrypted message: "
                << static_cast<std::string>(decrypted_message->value());
      break;
    }
    PumpOutgoingMessages(&client_session, stream.get());
  }

  LOG(INFO) << "Closing stream and exiting.";
  stream->WritesDone();
  grpc::Status status = stream->Finish();
  if (status.ok()) {
    LOG(INFO) << "RPC finished successfully.";
  } else {
    LOG(ERROR) << "RPC failed: " << status.error_code() << ": "
               << status.error_message();
    return 1;
  }
  return 0;
}
