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
#include "containers/sql/sqlite_adapter.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gemma/gemma.h"
#include "util/threading_context.h"

namespace confidential_federated_compute::fed_sql {

// Session configuration for running inference using gemma.cpp engine.
struct SessionGemmaCppConfiguration {
  std::string tokenizer_path;
  std::string model_weight_path;
};

// Session configuration for running inference using llama.cpp engine.
struct SessionLlamaCppConfiguration {
  std::string model_weight_path;
};

// Configuration of the per-client inference step, occurring before the
// per-client query step.
struct SessionInferenceConfiguration {
  fcp::confidentialcompute::InferenceInitializeConfiguration
      initialize_configuration;
  std::optional<SessionGemmaCppConfiguration> gemma_configuration;
  std::optional<SessionLlamaCppConfiguration> llama_configuration;
};

// An LLM model that can be invoked to run inference before the per-client query
// step.
class InferenceModel {
 public:
  absl::Status BuildModel(
      const SessionInferenceConfiguration& inference_configuration);
  absl::Status RunInference(
      std::vector<::tensorflow_federated::aggregation::Tensor>& columns);
  bool HasModel() const;
  const std::optional<SessionInferenceConfiguration>&
  GetInferenceConfiguration() const;

 private:
  struct NoModel {};
  struct GemmaCppModel {
    std::unique_ptr<::gcpp::Gemma> gemma_;
    std::unique_ptr<::gcpp::MatMulEnv> env_;
    std::unique_ptr<::gcpp::ThreadingContext> ctx_;
  };

  // Builds a gemma.cpp compatible model from the given Gemma config.
  // This function assumes that the model_ is already a GemmaCppModel.
  virtual void BuildGemmaCppModel(
      const SessionGemmaCppConfiguration& gemma_config);

  // Runs inference with the gemma.cpp model using given prompt over all rows
  // and returns a 1-D string tensor of results representing the output column.
  virtual absl::StatusOr<::tensorflow_federated::aggregation::Tensor>
  RunGemmaCppInference(
      const ::fcp::confidentialcompute::Prompt& prompt,
      const absl::flat_hash_map<
          std::string, absl::Span<const absl::string_view>>& column_values,
      const std::string& output_column_name);

  std::optional<SessionInferenceConfiguration> inference_configuration_;
  std::variant<NoModel, GemmaCppModel> model_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_MODEL_H_
