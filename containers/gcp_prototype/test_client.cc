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

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cc/containers/sdk/orchestrator_client.h"
#include "cc/oak_session/client_session.h"
#include "client_session_config.h"
#include "google/protobuf/text_format.h"
#include "grpcpp/grpcpp.h"
#include "policy.pb.h"
#include "proto/services/session_v1_service.grpc.pb.h"
#include "proto_util.h"
#include "session_utils.h"
#include "test_client.grpc.pb.h"
#include "verifier.h"

using gcp_prototype::AttestationPolicy;
using gcp_prototype::MyVerifier;
using gcp_prototype::TestClientGenerateRequest;
using gcp_prototype::TestClientGenerateResponse;
using gcp_prototype::TestClientService;
using ::oak::containers::sdk::OrchestratorClient;
using ::oak::services::OakSessionV1Service;
using ::oak::session::ClientSession;
using ::oak::session::SessionConfig;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

const int SERVER_PORT = 8000;
const int CLIENT_PORT = 8080;

ABSL_FLAG(std::string, server_address, "localhost", "Server address.");
ABSL_FLAG(std::string, policy_path, "",
          "Path to the AttestationPolicy textproto file.");
ABSL_FLAG(std::string, jwks_path, "",
          "Path to local JWKS file for offline verification.");
ABSL_FLAG(bool, dump_jwt, false, "Dump JWT payload (DEBUG).");

// Reads the policy file and parses it into the protobuf.
AttestationPolicy ReadPolicyOrDie() {
  std::string policy_path = absl::GetFlag(FLAGS_policy_path);
  if (policy_path.empty()) {
    LOG(FATAL) << "--policy_path must be specified.";
  }
  AttestationPolicy policy;
  gcp_prototype::ReadTextProtoOrDie(policy_path, &policy);
  return policy;
}

/**
 * @brief gRPC service implementation that acts as a secure bridge.
 *
 * This service listens for plain-text prompts from the untrusted host,
 * establishes a secure, attested session with the GCP backend, forwards
 * the prompt, and returns the decrypted response.
 *
 * Currently stateless: a new session is established for every request.
 */
class TestClientServiceImpl final : public TestClientService::Service {
 public:
  explicit TestClientServiceImpl(MyVerifier* verifier) : verifier_(verifier) {}

  grpc::Status Generate(grpc::ServerContext* context,
                        const TestClientGenerateRequest* request,
                        TestClientGenerateResponse* response) override {
    LOG(INFO) << "Received Generate request with prompt: " << request->prompt();

    using gcp_prototype::session_utils::ExchangeHandshakeMessages;
    using gcp_prototype::session_utils::PumpOutgoingMessages;

    // 1. Initialize Session & Connect
    LOG(INFO) << "Initializing Oak ClientSession via Rust.";
    // We use FFI here because the Oak ClientSession is implemented in Rust.
    // We pass the C++ verifier object and a callback function so that Rust can
    // delegate the actual JWT verification back to our C++ implementation.
    SessionConfig* client_config = create_client_session_config(
        static_cast<void*>(verifier_), &verify_jwt_f);

    absl::StatusOr<std::unique_ptr<ClientSession>> client_session_or =
        ClientSession::Create(client_config);
    if (!client_session_or.ok()) {
      LOG(ERROR) << "Failed to create ClientSession: "
                 << client_session_or.status();
      return grpc::Status(grpc::StatusCode::INTERNAL,
                          "Failed to create ClientSession");
    }
    ClientSession& client_session = **client_session_or;

    std::string server_target = absl::StrFormat(
        "%s:%d", absl::GetFlag(FLAGS_server_address), SERVER_PORT);
    LOG(INFO) << "Connecting to server at " << server_target;

    auto channel =
        grpc::CreateChannel(server_target, grpc::InsecureChannelCredentials());
    auto stub = OakSessionV1Service::NewStub(channel);
    grpc::ClientContext client_context;
    auto stream = stub->Stream(&client_context);

    // 2. Execute Flow: Handshake, send message, receive response.
    // The handshake will trigger the Rust session to call our C++ verification
    // callback.
    ExchangeHandshakeMessages(&client_session, stream.get());

    LOG(INFO) << "Client encrypting and sending application message.";
    CHECK_OK(client_session.Write(request->prompt()));
    CHECK_OK(PumpOutgoingMessages(&client_session, stream.get()));

    LOG(INFO) << "Waiting for server reply...";
    std::string final_response_text;
    bool got_response = false;

    while (true) {
      SessionResponse session_response;
      if (!stream->Read(&session_response)) {
        LOG(ERROR)
            << "Server closed stream while waiting for application reply.";
        break;
      }
      LOG(INFO) << "gRPC -> Oak: " << session_response.DebugString();
      CHECK_OK(client_session.PutIncomingMessage(session_response));
      auto decrypted_message = client_session.ReadToRustBytes();
      CHECK_OK(decrypted_message.status()) << "Failed to read from session";

      if (decrypted_message->has_value()) {
        final_response_text =
            static_cast<std::string>(decrypted_message->value());
        LOG(INFO) << "Client decrypted message: " << final_response_text;
        got_response = true;
        break;
      }
      CHECK_OK(PumpOutgoingMessages(&client_session, stream.get()));
    }

    LOG(INFO) << "Closing stream to GCP.";
    stream->WritesDone();
    grpc::Status status = stream->Finish();
    if (status.ok()) {
      LOG(INFO) << "RPC to GCP finished successfully.";
    } else {
      LOG(ERROR) << "RPC to GCP failed: " << status.error_code() << ": "
                 << status.error_message();
    }

    if (got_response) {
      response->set_response(final_response_text);
      return grpc::Status::OK;
    } else {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                          "Did not receive valid response from GCP");
    }
  }

 private:
  MyVerifier* verifier_;
};

void RunServer(MyVerifier* verifier) {
  std::string server_address = absl::StrFormat("[::]:%d", CLIENT_PORT);
  TestClientServiceImpl service(verifier);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Test Client Server listening on " << server_address;

  // Initialize orchestrator client
  OrchestratorClient orchestrator_client;
  // Notify host we are ready to receive requests.
  if (auto status = orchestrator_client.NotifyAppReady(); !status.ok()) {
    LOG(WARNING) << "Failed to notify host of readiness: " << status;
  }

  server->Wait();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // 1. Configure & Initialize Verifier ONCE at startup
  MyVerifier verifier;
  verifier.SetPolicy(ReadPolicyOrDie());
  verifier.SetDumpJwt(absl::GetFlag(FLAGS_dump_jwt));

  // Fetch JWKS (or load from file) and prepare Tink for verification.
  CHECK_OK(verifier.Initialize(absl::GetFlag(FLAGS_jwks_path)))
      << "Failed to initialize attestation verifier";

  RunServer(&verifier);

  return 0;
}
