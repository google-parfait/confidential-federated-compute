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
#include "containers/fed_sql/inference_model_helper.h"
#include "containers/sql/input.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gemma/gemma.h"
#include "include/llama.h"
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
  // The following param is currently not surfaced in the customer-facing API.
  int32_t n_gpu_layers = 0;
};

// Configuration of the per-client inference step, occurring before the
// per-client query step.
struct SessionInferenceConfiguration {
  fcp::confidentialcompute::InferenceInitializeConfiguration
      initialize_configuration;
  std::optional<SessionGemmaCppConfiguration> gemma_configuration;
  std::optional<SessionLlamaCppConfiguration> llama_configuration;
};

// An LLM model that can be invoked to run inference before the per-client
// query step.
class InferenceModel {
 public:
  absl::Status BuildModel(
      const SessionInferenceConfiguration& inference_configuration);
  // Runs inference on the given input. The input is expected to contain the
  // columns required by the inference task. The output column produced by
  // the inference model will be added to the input.
  absl::Status RunInference(sql::Input& input);
  bool HasModel() const;
  const std::optional<SessionInferenceConfiguration>&
  GetInferenceConfiguration() const;

 private:
  struct NoModel {};
  struct GemmaCppModel {
    std::unique_ptr<gcpp::Gemma> gemma_;
    std::unique_ptr<gcpp::MatMulEnv> env_;
    std::unique_ptr<gcpp::ThreadingContext> ctx_;
  };
  struct LlamaCppModel {
   private:
    // Define the custom deleter privately inside the struct
    struct LlamaModelDeleter {
      void operator()(llama_model* model) const {
        if (model) {
          llama_model_free(model);
        }
      }
    };

   public:
    std::unique_ptr<llama_model, LlamaModelDeleter> llama_;
    ;
  };

  // Builds a gemma.cpp compatible model from the given Gemma config.
  // This function assumes that the model_ is already a GemmaCppModel.
  virtual absl::Status BuildGemmaCppModel(
      const SessionGemmaCppConfiguration& gemma_config);

  // Builds a llama.cpp model from the given Llama config.
  // This function assumes that the model_ is already a LlamaModel.
  virtual absl::Status BuildLlamaCppModel(
      const SessionLlamaCppConfiguration& llama_config);

  // Runs inference with the gemma.cpp model using given prompt over all
  // rows and returns a 1-D string tensor of results representing the output
  // column.
  virtual absl::StatusOr<::tensorflow_federated::aggregation::Tensor>
  RunGemmaCppInference(const fcp::confidentialcompute::Prompt& prompt,
                       const sql::Input& input,
                       absl::Span<const size_t> input_column_indices,
                       const std::string& output_column_name);

  // Runs inference with the llama.cpp model using given prompt over all
  // rows and returns a 1-D string tensor of results representing the output
  // column.
  virtual absl::StatusOr<::tensorflow_federated::aggregation::Tensor>
  RunLlamaCppInference(const fcp::confidentialcompute::Prompt& prompt,
                       const sql::Input& input,
                       absl::Span<const size_t> input_column_indices,
                       const std::string& output_column_name);

  // Helpter function that runs inference with the llama.cpp model using
  // given prompt over a single row.
  absl::StatusOr<std::string> RunLlamaCppInferencePerRow(
      const std::string& combined_prompt, LlamaCppModel& llama_model,
      const llama_vocab* vocab);

  std::optional<SessionInferenceConfiguration> inference_configuration_;
  std::variant<NoModel, GemmaCppModel, LlamaCppModel> model_;
  InferencePromptProcessor prompt_processor_;
  InferenceOutputProcessor output_processor_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_MODEL_H_
