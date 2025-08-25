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
// Helper functions and classes for FedSQL unit tests.

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_MOCKS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_MOCKS_H_

#include <string>

#include "containers/fed_sql/inference_model.h"
#include "gmock/gmock.h"

namespace confidential_federated_compute::fed_sql::testing {

class MockInferenceModel : public InferenceModel {
 public:
  MOCK_METHOD(void, BuildGemmaModel,
              (const SessionGemmaConfiguration& gemma_config), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, RunGemmaInference,
              (const std::string& prompt, const absl::string_view& column_value,
               const std::string& column_name),
              (override));
};

}  // namespace confidential_federated_compute::fed_sql::testing
#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_MOCKS_H_
