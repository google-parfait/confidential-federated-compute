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

#ifndef INFERENCE_ENGINE_H_
#define INFERENCE_ENGINE_H_

#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "include/llama.h"
#include "inference.pb.h"

namespace gcp_prototype {

/**
 * @brief A thread-safe wrapper around llama.cpp for sequential LLM inference.
 *
 * This class encapsulates the state required to run a llama.cpp model,
 * including the loaded model weights, the active context (KV cache), and the
 * sampler. It provides a simplified, synchronous interface for generating
 * text from a prompt.
 *
 * Thread Safety: Internally synchronized with a mutex. Concurrent calls to
 * Infer() will block until the previous inference is complete.
 */
class InferenceEngine {
 public:
  /**
   * @brief Initializes the engine by loading the model from the specified path.
   *
   * This operation is I/O intensive and slow. It should be called once at
   * application startup.
   *
   * @param model_path Filesystem path to the GGUF model file.
   * @param gpu_layers Number of layers to offload to GPU (0 for CPU only).
   * @return A unique_ptr to the initialized engine, or an error status.
   */
  static absl::StatusOr<std::unique_ptr<InferenceEngine>> Create(
      const std::string& model_path, int gpu_layers);

  ~InferenceEngine();

  /**
   * @brief Performs full inference on the given prompt.
   *
   * This method blocks until the complete response is generated.
   * It handles applying the appropriate chat template (currently hardcoded
   * for Gemma), tokenization, and the generation loop.
   *
   * @param prompt The user's input text.
   * @return The generated response string, or an error status.
   */
  absl::StatusOr<std::string> Infer(const std::string& prompt);

  /**
   * @brief Performs batched inference on a list of prompts.
   *
   * This method processes multiple prompts in parallel using llama.cpp's
   * batch decoding capabilities.
   *
   * @param request The proto containing the list of prompts and parameters.
   * @return A response proto with results for every prompt.
   */
  absl::StatusOr<BatchedInferenceResponse> InferBatch(
      const BatchedInferenceRequest& request);

 private:
  // Private constructor used by Create().
  InferenceEngine(llama_model* model, const llama_vocab* vocab);

  // Helper to tokenize a prompt (including chat template application).
  absl::StatusOr<std::vector<llama_token>> Tokenize(const std::string& prompt);

  llama_model* model_;        // Owned by this class.
  const llama_vocab* vocab_;  // Owned by model_.

  absl::Mutex mutex_;
  // Context (KV cache) and sampler are reused across requests to save
  // allocation time, but are reset at the start of each InferInternal() call
  // to ensure stateless request handling.
  llama_context* ctx_ ABSL_GUARDED_BY(mutex_) = nullptr;
  llama_sampler* sampler_ ABSL_GUARDED_BY(mutex_) = nullptr;

  // Reusable batch structure for llama.cpp to avoid frequent allocations.
  llama_batch batch_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gcp_prototype

#endif  // INFERENCE_ENGINE_H_
