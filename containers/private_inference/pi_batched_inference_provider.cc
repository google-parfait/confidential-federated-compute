
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
#include "pi_batched_inference_provider.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "pi_client.h"

namespace confidential_federated_compute::private_inference {

const int DEFAULT_HOST_PROXY_PORT_WHEN_UNSPECIFIED = 8000;

PiBatchedInferenceEngine::PiBatchedInferenceEngine(
    std::unique_ptr<PiClient> pi_client)
    : pi_client_(std::move(pi_client)) {}

std::vector<absl::StatusOr<std::string>>
PiBatchedInferenceEngine::DoBatchedInference(std::vector<std::string> prompts) {
  std::vector<absl::StatusOr<std::string>> results;
  results.reserve(prompts.size());
  for (const std::string& prompt : prompts) {
    results.push_back(pi_client_->Generate(prompt));
  }
  return results;
}

PiBatchedInferenceProvider::PiBatchedInferenceProvider(
    std::string server_address,
    ::private_inference::proto::FeatureName feature_name)
    : server_address_(std::move(server_address)), feature_name_(feature_name) {}

std::shared_ptr<
    ::confidential_federated_compute::batched_inference::BatchedInferenceEngine>
PiBatchedInferenceProvider::GetEngineForInferenceConfig(
    const fcp::confidentialcompute::InferenceConfiguration& config) {
  int port = DEFAULT_HOST_PROXY_PORT_WHEN_UNSPECIFIED;
  if (config.has_proxy_config() &&
      config.proxy_config().host_proxy_port() > 0) {
    port = config.proxy_config().host_proxy_port();
  }
  // The port number is taken from the InferenceConfiguration received from
  // the collaborator's server. This assumes server_address_ is just an IP
  // or hostname without a port.
  std::string server_target = absl::StrFormat("%s:%d", server_address_, port);
  absl::StatusOr<std::unique_ptr<PiClient>> pi_client =
      CreatePiClient(server_target, feature_name_);
  if (!pi_client.ok()) {
    LOG(ERROR) << "Failed to create PiClient: " << pi_client.status();
    return nullptr;
  }
  return std::make_shared<PiBatchedInferenceEngine>(
      std::move(pi_client.value()));
}

}  // namespace confidential_federated_compute::private_inference
