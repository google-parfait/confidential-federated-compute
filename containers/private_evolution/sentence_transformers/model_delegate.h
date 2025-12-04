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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_MODEL_DELEGATE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_MODEL_DELEGATE_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace confidential_federated_compute::sentence_transformers {

// Interface for ModelDelegate.
// This interface is useful for separating core business logic with the rest of
// the Session logic. Each Session is expected to hold its own instance of
// ModelDelegate. This interface is also useful for injecting fake
// implementations into tests.
class ModelDelegate {
 public:
  virtual ~ModelDelegate() = default;
  virtual void InitializeRuntime() {};
  virtual void FinalizeRuntime() {};
  virtual bool InitializeModel(absl::string_view artifact_path) = 0;
  virtual absl::StatusOr<std::vector<std::vector<float>>> GenerateEmbeddings(
      const std::vector<std::string>& inputs,
      std::optional<std::string> prompt) = 0;
};

// Factory class for creating ModelDelegate.
class ModelDelegateFactory {
 public:
  virtual std::unique_ptr<ModelDelegate> Create() = 0;
};

}  // namespace confidential_federated_compute::sentence_transformers

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_MODEL_DELEGATE_H_
