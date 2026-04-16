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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_INFERENCE_PI_BATCHED_INFERENCE_PROVIDER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_INFERENCE_PI_BATCHED_INFERENCE_PROVIDER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "containers/batched_inference/batched_inference_engine.h"
#include "pi_client.h"

namespace confidential_federated_compute::private_inference {

class PiBatchedInferenceEngine : public ::confidential_federated_compute::
                                     batched_inference::BatchedInferenceEngine {
 public:
  explicit PiBatchedInferenceEngine(std::unique_ptr<PiClient> pi_client);
  ~PiBatchedInferenceEngine() override = default;

  std::vector<absl::StatusOr<std::string>> DoBatchedInference(
      std::vector<std::string> prompts) override;

 private:
  std::unique_ptr<PiClient> pi_client_;
};

class PiBatchedInferenceProvider
    : public ::confidential_federated_compute::batched_inference::
          BatchedInferenceEngineProvider {
 public:
  PiBatchedInferenceProvider(
      std::string server_address,
      ::private_inference::proto::FeatureName feature_name);
  ~PiBatchedInferenceProvider() override = default;

  std::shared_ptr<::confidential_federated_compute::batched_inference::
                      BatchedInferenceEngine>
  GetEngineForInferenceConfig(
      const fcp::confidentialcompute::InferenceConfiguration& config) override;

 private:
  std::string server_address_;
  ::private_inference::proto::FeatureName feature_name_;
};

}  // namespace confidential_federated_compute::private_inference

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_INFERENCE_PI_BATCHED_INFERENCE_PROVIDER_H_