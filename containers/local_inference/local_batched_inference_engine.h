// Copyright 2026 Google LLC.
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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_LOCAL_INFERENCE_LOCAL_BATCHED_INFERENCE_ENGINE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_LOCAL_INFERENCE_LOCAL_BATCHED_INFERENCE_ENGINE_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "containers/common/inference/batched_inference_engine.h"
#include "containers/common/local_inference/inference_model.h"
#include "containers/fns/fn_factory.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"

namespace confidential_federated_compute::local_inference {

// A BatchedInferenceEngine backed by an in-process gemma.cpp or llama.cpp
// model.
//
// This is a thin adapter: it does no templating, no parsing, and no row
// expansion. BatchedInferenceFn owns all of that, exactly as it does for the
// remote-backend engines in containers/private_inference and containers/gcp.
//
// This class is thread-safe.
class LocalBatchedInferenceEngine : public ::confidential_federated_compute::
                                        inference::BatchedInferenceEngine {
 public:
  explicit LocalBatchedInferenceEngine(
      std::shared_ptr<InferenceModel> inference_model);
  ~LocalBatchedInferenceEngine() override = default;

  std::vector<absl::StatusOr<std::string>> DoBatchedInference(
      std::vector<std::string> prompts) override;

 private:
  std::shared_ptr<InferenceModel> inference_model_;
};

// Builds and hands out the shared model used by every BatchedInferenceFn.
//
// Only MODEL_MODE_SHARED_MODEL is supported: the weights are loaded once at
// StreamInitialize and shared read-only across all BatchedInferenceFn
// instances. MODEL_MODE_MODEL_PER_SESSION is rejected.
class LocalBatchedInferenceEngineProvider
    : public ::confidential_federated_compute::inference::
          BatchedInferenceEngineProvider {
 public:
  LocalBatchedInferenceEngineProvider() = default;
  ~LocalBatchedInferenceEngineProvider() override = default;

  absl::StatusOr<std::shared_ptr<
      ::confidential_federated_compute::inference::BatchedInferenceEngine>>
  GetEngineForInferenceConfig(
      const fcp::confidentialcompute::InferenceInitializeConfiguration&
          init_config,
      const fns::WriteConfigurationMap& write_configuration_map) override;
};

}  // namespace confidential_federated_compute::local_inference

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_LOCAL_INFERENCE_LOCAL_BATCHED_INFERENCE_ENGINE_H_
