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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_LOCAL_INFERENCE_INFERENCE_MODEL_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_LOCAL_INFERENCE_INFERENCE_MODEL_H_

#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gemma/gemma.h"
#include "include/llama.h"
#include "ops/matmul.h"
#include "util/threading_context.h"

namespace confidential_federated_compute::local_inference {

// Holds Gemma model weights (read-only after init) along with its
// ThreadingContext and MatMulEnv, whose access is serialized via mutex.
struct GemmaCppModel {
  std::unique_ptr<gcpp::Gemma> gemma;
  std::unique_ptr<gcpp::ThreadingContext> ctx;
  std::unique_ptr<gcpp::MatMulEnv> env;
  // Serializes GenerateBatch calls. This is technically redundant/inefficient
  // since we're mostly using a shared read-only model in practice, and with
  // such a shared read-only model we could in fact support concurrent
  // inferences on multiple threads. We should consider improving this
  // implementation.
  absl::Mutex mutex;
};

// Holds LlamaCppModel resources.
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
};

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
  // Optional pre-built Gemma model (read-only weights) for reuse across
  // sessions. If null, BuildModel builds a new GemmaCppModel.
  std::shared_ptr<GemmaCppModel> shared_gemma_model;
};

// An LLM model that can be invoked to generate text from already-rendered
// prompts.
//
// This class is a pure generation engine: it owns model weights and turns
// prompts into raw model output. It deliberately knows nothing about prompt
// templating, output parsing, or tabular data. Callers that need those
// behaviors layer them on top (see fed_sql::InferenceRunner for the tabular
// wrapper, and BatchedInferenceFn for the batched-inference pipeline).
//
// InferenceModel is thread-safe when a GemmaCppModel is used (all
// GenerateBatch calls will effectively be serialized). It is not thread-safe
// when used with a LlamaCppModel.
class InferenceModel {
 public:
  virtual ~InferenceModel() = default;

  InferenceModel(InferenceModel&&) = default;
  InferenceModel& operator=(InferenceModel&&) = default;

  InferenceModel(const InferenceModel&) = delete;
  InferenceModel& operator=(const InferenceModel&) = delete;

  static absl::StatusOr<InferenceModel> BuildModel(
      const SessionInferenceConfiguration& inference_configuration);
  // Runs the model over already-rendered prompts. Performs no templating, no
  // parsing, and no row expansion: returns exactly one result per input
  // prompt, in the same order. A failure that affects only part of the input
  // is reported per entry rather than failing the whole call.
  //
  // May chunk internally by RuntimeConfig.max_batch_size (only the
  // GemmaCppModel model code path supports it).
  //
  // NOTE: This inner chunking is needed for fed_sql::InferenceRunner, which
  // passes all prompts for a blob in a single call and relies on GenerateBatch
  // to bound how many prompts (and thus concurrent KVCache instances in
  // gemma.cpp) are active at once. When driven by BatchedInferenceFn, the
  // caller has already capped `prompts` at max_batch_size, so the chunk loop
  // runs once.
  virtual std::vector<absl::StatusOr<std::string>> GenerateBatch(
      absl::Span<const std::string> prompts);
  const SessionInferenceConfiguration& GetInferenceConfiguration() const;
  std::shared_ptr<GemmaCppModel> GetGemmaCppModel() const;

 protected:
  InferenceModel(
      SessionInferenceConfiguration inference_configuration,
      std::variant<std::shared_ptr<GemmaCppModel>, LlamaCppModel> model);

 private:
  // Builds a gemma.cpp compatible model from the given Gemma config.
  static absl::StatusOr<std::shared_ptr<GemmaCppModel>> BuildGemmaCppModel(
      const SessionInferenceConfiguration& inference_configuration);

  // Builds a llama.cpp model from the given Llama config.
  static absl::StatusOr<LlamaCppModel> BuildLlamaCppModel(
      const SessionLlamaCppConfiguration& llama_config);

  // Generates text with the gemma.cpp model for each of the given prompts,
  // processing them in chunks of RuntimeConfig.max_batch_size. Returns one
  // result per prompt, in order. A chunk that fails yields a failed status for
  // each of its own entries only; other chunks are unaffected.
  std::vector<absl::StatusOr<std::string>> GenerateBatchGemmaCpp(
      absl::Span<const std::string> prompts);

  // Generates text with the llama.cpp model for each of the given prompts, one
  // prompt at a time. Returns one result per prompt, in order.
  std::vector<absl::StatusOr<std::string>> GenerateBatchLlamaCpp(
      absl::Span<const std::string> prompts);

  // Helpter function that runs inference with the llama.cpp model using
  // given prompt over a single row.
  absl::StatusOr<std::string> RunLlamaCppInferencePerRow(
      const std::string& combined_prompt, LlamaCppModel& llama_model,
      const llama_vocab* vocab);

  SessionInferenceConfiguration inference_configuration_;
  std::variant<std::shared_ptr<GemmaCppModel>, LlamaCppModel> model_;
};

}  // namespace confidential_federated_compute::local_inference

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_LOCAL_INFERENCE_INFERENCE_MODEL_H_
