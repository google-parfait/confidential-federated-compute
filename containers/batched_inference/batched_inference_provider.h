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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINTERS_BATCHED_INFERENCE_BATCHED_INFERENCE_PROVIDER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINTERS_BATCHED_INFERENCE_BATCHED_INFERENCE_PROVIDER_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace confidential_federated_compute::batched_inference {

class BatchedInferenceProvider {
 public:
  virtual ~BatchedInferenceProvider() {}

  virtual std::vector<absl::StatusOr<std::string>> DoBatchedInference(
      std::vector<std::string> prompts) = 0;
};

}  // namespace confidential_federated_compute::batched_inference

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINTERS_BATCHED_INFERENCE_BATCHED_INFERENCE_PROVIDER_H_
