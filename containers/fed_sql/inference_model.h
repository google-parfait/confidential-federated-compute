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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_MODEL_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_MODEL_H_

#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gemma/gemma.h"

namespace confidential_federated_compute::fed_sql {

struct SessionGemmaConfiguration {
  std::string tokenizer_path;
  std::string model_weight_path;
};
// Configuration of the per-client inference step, occurring before the
// per-client query step.
struct SessionInferenceConfiguration {
  fcp::confidentialcompute::InferenceInitializeConfiguration
      initialize_configuration;
  std::optional<SessionGemmaConfiguration> gemma_configuration;
};
enum class ModelType : int {
  kNone = 0,
  kGemma = 1,
};
// An LLM model that can be invoked to run inference before the per-client query
// step.
class InferenceModel {
 public:
  absl::Status BuildModel(
      const SessionInferenceConfiguration& inference_configuration);
  absl::Status RunInference(
      std::vector<::confidential_federated_compute::sql::TensorColumn>&
          columns);
  bool HasModel() const;
  const std::optional<SessionInferenceConfiguration>&
  GetInferenceConfiguration() const;

 private:
  struct NoModel {};
  struct GemmaModel {
    std::unique_ptr<::gcpp::Gemma> gemma_;
    std::unique_ptr<::gcpp::NestedPools> pools_;
    std::unique_ptr<::gcpp::MatMulEnv> env_;
  };

  // Builds a Gemma model from the given model info and gemma config.
  // This function assumes that the model_ is already a GemmaModel.
  virtual void BuildGemmaModel(const ::gcpp::ModelInfo& model_info,
                               const SessionGemmaConfiguration& gemma_config);

  virtual absl::StatusOr<std::string> RunGemmaInference(
      const std::string& prompt, const absl::string_view& column_value,
      const std::string& column_name);
  ModelType GetModelType() const;

  absl::Mutex mutex_;
  std::optional<SessionInferenceConfiguration> inference_configuration_;
  std::variant<NoModel, GemmaModel> model_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_MODEL_H_