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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_INFERENCE_BATCHED_INFERENCE_ENGINE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_INFERENCE_BATCHED_INFERENCE_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "containers/fns/fn_factory.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"

namespace confidential_federated_compute::inference {

class BatchedInferenceEngine {
 public:
  virtual ~BatchedInferenceEngine() {}

  virtual std::vector<absl::StatusOr<std::string>> DoBatchedInference(
      std::vector<std::string> prompts) = 0;
};

class BatchedInferenceEngineProvider {
 public:
  virtual ~BatchedInferenceEngineProvider() {}

  // Returns the engine to use for the given initialization configuration.
  //
  // `write_configuration_map` maps configuration ids to the on-disk paths of
  // the blobs streamed during StreamInitialize, which is how backends that
  // load model weights in-process find them.
  virtual absl::StatusOr<std::shared_ptr<BatchedInferenceEngine>>
  GetEngineForInferenceConfig(
      const fcp::confidentialcompute::InferenceInitializeConfiguration&
          init_config,
      const fns::WriteConfigurationMap& write_configuration_map) = 0;
};

}  // namespace confidential_federated_compute::inference

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_INFERENCE_BATCHED_INFERENCE_ENGINE_H_
