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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_BATCHED_INFERENCE_ENGINE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_BATCHED_INFERENCE_ENGINE_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "batched_inference.pb.h"

namespace confidential_federated_compute::gcp {

// FUTURE WORK(b/452094015): Potentially merge with the
// BatchedInferenceProvider.
class BatchedInferenceEngine {
 public:
  virtual ~BatchedInferenceEngine() {}

  /**
   * @brief Performs batched inference on a list of prompts.
   *
   * This method processes multiple prompts in parallel.
   *
   * @param request The proto containing the list of prompts and parameters.
   * @return A response proto with results for every prompt.
   */
  virtual absl::StatusOr<BatchedInferenceResponse> DoBatchedInference(
      const BatchedInferenceRequest& request) = 0;
};

}  // namespace confidential_federated_compute::gcp

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_BATCHED_INFERENCE_ENGINE_H_
