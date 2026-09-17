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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_LOCAL_INFERENCE_TESTING_MOCKS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_LOCAL_INFERENCE_TESTING_MOCKS_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "containers/common/local_inference/inference_model.h"
#include "gmock/gmock.h"

namespace confidential_federated_compute::local_inference::testing {

class MockInferenceModel : public InferenceModel {
 public:
  MockInferenceModel()
      : InferenceModel(SessionInferenceConfiguration(),
                       std::make_shared<GemmaCppModel>()) {}
  explicit MockInferenceModel(
      SessionInferenceConfiguration config,
      std::variant<std::shared_ptr<GemmaCppModel>, LlamaCppModel> model =
          std::make_shared<GemmaCppModel>())
      : InferenceModel(std::move(config), std::move(model)) {}

  MOCK_METHOD(std::vector<absl::StatusOr<std::string>>, GenerateBatch,
              (absl::Span<const std::string> prompts), (override));
};

}  // namespace confidential_federated_compute::local_inference::testing

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_LOCAL_INFERENCE_TESTING_MOCKS_H_
