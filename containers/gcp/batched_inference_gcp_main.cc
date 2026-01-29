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
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "attestation_token_provider.h"
#include "batched_inference.pb.h"
#include "batched_inference_engine.h"
#include "llama_cpp_batched_inference_engine.h"
#include "server.h"

// Flag to select the attestation provider implementation.
ABSL_FLAG(std::string, attestation_provider, "ita",
          "Which attestation provider to use: 'gca' or 'ita'.");

// Flag to specify the path to the model weights (GGUF file).
ABSL_FLAG(std::string, model_path, "/saved_model/gemma-3-270m-it-q4_k_m.gguf",
          "Path to the model weights (GGUF file).");

ABSL_FLAG(int32_t, gpu_layers, 0,
          "Number of layers to offload to GPU. Set to 0 for CPU only, or a "
          "large number (e.g., 999) to offload all layers.");

namespace confidential_federated_compute::gcp {
namespace {

// Default server port.
const int PORT = 8000;

void RunServer() {
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

  std::unique_ptr<AttestationTokenProvider> attestation_token_provider =
      CreateAttestationTokenProvider(provider_type);

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  int32_t gpu_layers = absl::GetFlag(FLAGS_gpu_layers);

  LOG(INFO) << "Initializing inference engine... ";
  LOG(INFO) << "  Model: " << model_path;
  LOG(INFO) << "  GPU Layers: " << gpu_layers;

  absl::StatusOr<std::unique_ptr<BatchedInferenceEngine>> inference_engine_or =
      CreateLlamaCppBatchedInferenceEngine(model_path, gpu_layers);
  if (!inference_engine_or.ok()) {
    LOG(FATAL) << "Failed to initialize inference engine: "
               << inference_engine_or.status();
  }
  std::unique_ptr<BatchedInferenceEngine> inference_engine =
      std::move(*inference_engine_or);

  LOG(INFO) << "Inference engine initialized successfully.";

  Server::RequestHandler request_handler =
      [engine = std::move(inference_engine)](
          std::string request) mutable -> absl::StatusOr<std::string> {
    // Parse the Batched Request Proto
    BatchedInferenceRequest batch_request;
    if (!batch_request.ParseFromString(request)) {
      return absl::InternalError(
          "Failed to parse BatchedInferenceRequest proto.");
    }
    LOG(INFO) << "Received batch with " << batch_request.requests_size()
              << " requests.";
    auto start_time = std::chrono::high_resolution_clock::now();
    absl::StatusOr<BatchedInferenceResponse> response_or =
        engine->DoBatchedInference(batch_request);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    if (!response_or.ok()) {
      return absl::InternalError(absl::StrCat("Batch inference failed: ",
                                              response_or.status().ToString()));
    }
    LOG(INFO) << "Batch processing complete (" << elapsed.count() << "s).";
    std::string response_payload;
    if (!response_or->SerializeToString(&response_payload)) {
      return absl::InternalError(
          "Failed to serialize BatchedInferenceResponse.");
    }
    return response_payload;
  };

  absl::StatusOr<std::unique_ptr<Server>> server_or = CreateServer(
      PORT, std::move(attestation_token_provider), std::move(request_handler));
  if (!server_or.ok()) {
    LOG(FATAL) << "Failed to create the server: " << server_or.status();
  }

  server_or.value()->Wait();
}

}  // namespace
}  // namespace confidential_federated_compute::gcp

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  confidential_federated_compute::gcp::RunServer();
  return 0;
}
