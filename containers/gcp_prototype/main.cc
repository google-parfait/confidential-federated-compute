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

// gRPC server implementation using Oak Noise sessions for secure communication
// and GCP Confidential Space attestation.

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"  // For Base64Escape
#include "absl/strings/str_format.h"
#include "attestation.h"
#include "cc/ffi/rust_bytes.h"
#include "cc/oak_session/server_session.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "inference_engine.h"
#include "proto/services/session_v1_service.grpc.pb.h"
#include "server_session_config.h"
#include "session_utils.h"
#include "tink/config/tink_config.h"
#include "tink/signature/signature_config.h"

// Using declarations for convenience.
using ::oak::services::OakSessionV1Service;
using ::oak::session::ServerSession;
using ::oak::session::SessionConfig;
using ::oak::session::SigningKeyHandle;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

using ::gcp_prototype::AttestationTokenProvider;
using ::gcp_prototype::CreateTokenProvider;
using ::gcp_prototype::InferenceEngine;
using ::gcp_prototype::ProviderType;

// Default server port.
const int PORT = 8000;

// Flag to select the attestation provider implementation.
ABSL_FLAG(std::string, attestation_provider, "ita",
          "Which attestation provider to use: 'gca' or 'ita'.");

// Flag to specify the path to the model weights (GGUF file).
ABSL_FLAG(std::string, model_path, "/saved_model/gemma-3-270m-it-q4_k_m.gguf",
          "Path to the model weights (GGUF file).");

ABSL_FLAG(int32_t, gpu_layers, 0,
          "Number of layers to offload to GPU. Set to 0 for CPU only, or a "
          "large number (e.g., 999) to offload all layers.");
/**
 * @brief Implementation of the Oak Session gRPC service.
 * Handles a single client stream connection.
 */
class OakSessionV1ServiceImpl final : public OakSessionV1Service::Service {
 public:
  explicit OakSessionV1ServiceImpl(
      std::unique_ptr<AttestationTokenProvider> provider,
      InferenceEngine* inference_engine)
      : token_provider_(std::move(provider)),
        inference_engine_(inference_engine) {
    CHECK(token_provider_ != nullptr) << "Token provider cannot be null";
    CHECK(inference_engine_ != nullptr) << "Inference engine cannot be null";
  }

  grpc::Status Stream(grpc::ServerContext* context,
                      grpc::ServerReaderWriter<SessionResponse, SessionRequest>*
                          stream) override {
    LOG(INFO) << "gRPC stream started. Generating key pair via Rust FFI.";

    // 1. Generate ephemeral key pair in Rust.
    std::vector<unsigned char> public_key_bytes(128);
    SigningKeyHandle* private_key_handle = nullptr;

    int public_key_len = generate_key_pair(
        public_key_bytes.data(), public_key_bytes.size(), &private_key_handle);

    if (public_key_len < 0 || private_key_handle == nullptr) {
      LOG(ERROR) << "Failed to generate key pair via Rust FFI (return code "
                 << public_key_len << ").";
      return grpc::Status(grpc::StatusCode::INTERNAL, "Key generation failed");
    }

    public_key_bytes.resize(public_key_len);
    LOG(INFO) << "Generated key pair. Public key size: " << public_key_len;

    // 2. Create nonce from public key and fetch attestation token.
    std::string public_key_str(public_key_bytes.begin(),
                               public_key_bytes.end());
    std::string nonce = absl::Base64Escape(public_key_str);
    LOG(INFO) << "Using public key nonce (Base64): " << nonce;

    absl::StatusOr<std::string> token_or =
        token_provider_->GetAttestationToken(nonce);
    if (!token_or.ok()) {
      LOG(ERROR) << "Failed to get attestation token: " << token_or.status();
      return grpc::Status(grpc::StatusCode::INTERNAL,
                          absl::StrCat("Attestation token fetch failed: ",
                                       token_or.status().ToString()));
    }
    std::string token = *token_or;
    LOG(INFO) << "Successfully fetched attestation token (size "
              << token.length() << ").";

    // 3. Configure Oak Session with the token and private key handle.
    LOG(INFO)
        << "Passing token and key handle to Rust to create SessionConfig.";
    SessionConfig* server_config = create_server_session_config(
        token.c_str(), token.length(), private_key_handle);

    if (server_config == nullptr) {
      LOG(FATAL) << "Rust create_server_session_config returned null.";
      return grpc::Status(grpc::StatusCode::INTERNAL,
                          "Session config creation failed");
    }

    absl::StatusOr<std::unique_ptr<ServerSession>> server_session_or =
        ServerSession::Create(server_config);
    CHECK_OK(server_session_or.status()) << "Failed to create ServerSession";
    std::unique_ptr<ServerSession> server_session =
        std::move(*server_session_or);

    LOG(INFO) << "ServerSession created, entering message processing loop.";

    // Use the shared PumpOutgoingMessages function.
    using gcp_prototype::session_utils::PumpOutgoingMessages;

    // 4. Main loop: Read requests, process via Oak, send responses.
    while (true) {
      SessionRequest request;
      if (!stream->Read(&request)) {
        LOG(INFO) << "Client closed the stream.";
        break;
      }
      LOG(INFO) << "gRPC -> Oak: " << request.DebugString();
      absl::Status put_status = server_session->PutIncomingMessage(request);
      CHECK_OK(put_status) << "Failed to process incoming message";

      // Send any handshake or application data generated by Oak.
      CHECK_OK(PumpOutgoingMessages(server_session.get(), stream));

      // Check if we have decrypted application data.
      auto decrypted_message = server_session->ReadToRustBytes();
      if (!decrypted_message.ok()) {
        if (decrypted_message.status().code() != absl::StatusCode::kInternal) {
          LOG(FATAL) << "Failed to read from session: "
                     << decrypted_message.status();
        }
        continue;
      }

      if (!decrypted_message->has_value()) {
        continue;
      }

      std::string decrypted_request =
          static_cast<std::string>(decrypted_message->value());
      LOG(INFO) << "Server decrypted application message: \""
                << decrypted_request << "\"";

      // Run inference.
      LOG(INFO) << "Running inference...";

      auto start_time =
          std::chrono::high_resolution_clock::now();  // START TIMER

      absl::StatusOr<std::string> inference_result =
          inference_engine_->Infer(decrypted_request);

      auto end_time = std::chrono::high_resolution_clock::now();  // END TIMER
      std::chrono::duration<double> elapsed = end_time - start_time;

      LOG(INFO) << "Inference completed in " << elapsed.count() << " seconds.";

      std::string response_payload;
      if (inference_result.ok()) {
        response_payload = *inference_result;
        LOG(INFO) << "Inference successful.";
      } else {
        LOG(ERROR) << "Inference failed: " << inference_result.status();
        response_payload = "Error: Inference failed.";
      }

      LOG(INFO) << "Server encrypting and sending application reply.";
      absl::Status write_status = server_session->Write(response_payload);
      CHECK_OK(write_status) << "Failed to write reply message";

      CHECK_OK(PumpOutgoingMessages(server_session.get(), stream));
    }

    LOG(INFO) << "gRPC stream finished.";
    return grpc::Status::OK;
  }

 private:
  std::unique_ptr<AttestationTokenProvider> token_provider_;
  InferenceEngine* inference_engine_;
};

/**
 * @brief Sets up and runs the gRPC server.
 */
void RunServer(InferenceEngine* inference_engine) {
  std::string server_address = absl::StrFormat("0.0.0.0:%d", PORT);

  // Select and create the appropriate attestation provider based on flag.
  std::unique_ptr<AttestationTokenProvider> token_provider;
  std::string provider_flag = absl::GetFlag(FLAGS_attestation_provider);

  ProviderType provider_type;
  if (provider_flag == "gca") {
    provider_type = ProviderType::kGca;
  } else if (provider_flag == "ita") {
    provider_type = ProviderType::kIta;
  } else {
    LOG(ERROR) << "Invalid --attestation_provider flag value: '"
               << provider_flag << "'. Use 'gca' or 'ita'.";
    exit(1);
  }
  token_provider = CreateTokenProvider(provider_type);

  OakSessionV1ServiceImpl service(std::move(token_provider), inference_engine);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;
  server->Wait();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // Initialize Tink for potential future server-side crypto needs.
  auto status = crypto::tink::TinkConfig::Register();
  CHECK_OK(status) << "Failed to register TinkConfig";
  status = crypto::tink::SignatureConfig::Register();
  CHECK_OK(status) << "Failed to register Tink SignatureConfig";

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  int32_t gpu_layers = absl::GetFlag(FLAGS_gpu_layers);
  LOG(INFO) << "Initializing inference engine... ";
  LOG(INFO) << "  Model: " << model_path;
  LOG(INFO) << "  GPU Layers: " << gpu_layers;
  auto engine_or = InferenceEngine::Create(model_path, gpu_layers);
  if (!engine_or.ok()) {
    LOG(FATAL) << "Failed to initialize inference engine: "
               << engine_or.status();
  }
  std::unique_ptr<InferenceEngine> inference_engine = std::move(*engine_or);
  LOG(INFO) << "Inference engine initialized successfully.";

  RunServer(inference_engine.get());

  return 0;
}
