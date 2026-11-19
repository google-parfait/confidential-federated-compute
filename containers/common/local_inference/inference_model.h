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

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gemma/gemma.h"
#include "gemma/kv_cache.h"
#include "include/llama.h"
#include "ops/matmul.h"
#include "util/threading_context.h"

namespace confidential_federated_compute::local_inference {

// Holds Gemma model weights (read-only after init) along with its
// ThreadingContext, MatMulEnv and KV cache, whose access is serialized via
// mutex.
struct GemmaCppModel {
  std::unique_ptr<gcpp::Gemma> gemma;
  std::unique_ptr<gcpp::ThreadingContext> ctx;
  std::unique_ptr<gcpp::MatMulEnv> env;
  // Serializes GenerateBatch calls, which share `env` and the mutable KV cache
  // state below.
  absl::Mutex mutex;
  // KV cache reused by every prompt processed with this model, including
  // across InferenceModel instances sharing this GemmaCppModel (prefix
  // caching). Declared after `ctx` since it uses its allocator. Null until the
  // first prompt is processed.
  //
  // TODO: Use multiple KV caches (e.g. one per inference task) so that prompts
  // with different prefixes, such as prompts of interleaved inference tasks,
  // don't evict each other's cached prefix. Each prompt would then be routed
  // to the cache sharing the longest token prefix with it, possibly with a
  // small number of runners processing prompts in parallel.
  std::unique_ptr<gcpp::KVCache> kv_cache ABSL_GUARDED_BY(mutex);
  // Tokens whose KV entries at positions [0, cached_tokens.size()) in
  // `kv_cache` are valid. A prompt that starts with the same tokens reuses
  // those entries instead of prefilling them again.
  std::vector<int> cached_tokens ABSL_GUARDED_BY(mutex);
};

// Returns the number of leading `prompt_tokens` whose KV entries can be reused
// from a KV cache holding the entries of `cached_tokens`: the length of their
// longest common prefix, capped at prompt_tokens.size() - 1 since at least one
// prompt token must be processed to start generation.
size_t ReusablePrefixLength(absl::Span<const int> cached_tokens,
                            absl::Span<const int> prompt_tokens);

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
//
// With a GemmaCppModel, prompts are processed one at a time and prefix caching
// is used: each prompt reuses the KV entries of the longest token prefix it
// shares with the previously processed prompt, and only that immediately
// preceding prompt. Prefixes shared with non-immediately preceding prompts are
// not reused.
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
  // The GemmaCppModel code path processes the prompts one at a time,
  // regardless of RuntimeConfig.max_batch_size.
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
  // one prompt at a time. Returns one result per prompt, in order.
  std::vector<absl::StatusOr<std::string>> GenerateBatchGemmaCpp(
      absl::Span<const std::string> prompts);

  // Generates text with the gemma.cpp model for a single prompt, reusing the
  // KV entries of the longest token prefix it shares with the previously
  // processed prompt.
  static absl::StatusOr<std::string> GenerateGemmaCpp(
      GemmaCppModel& gemma_model, const std::string& prompt,
      size_t max_output_tokens, float temperature);

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
