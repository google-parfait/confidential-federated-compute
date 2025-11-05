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

#include "inference_engine.h"

#include <iostream>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace gcp_prototype {

namespace {
// Hardcoded parameters for this prototype step.
constexpr int kGpuLayers = 0;  // CPU only for step 1
constexpr int kMaxTokensToPredict = 1024;

// Custom log callback to suppress verbose output from llama.cpp
static void LlamaNoOpLogger(ggml_log_level level, const char* text,
                            void* user_data) {
  if (level == GGML_LOG_LEVEL_ERROR) {
    LOG(ERROR) << "llama.cpp: " << text;
  }
}
}  // namespace

InferenceEngine::InferenceEngine(llama_model* model, const llama_vocab* vocab)
    : model_(model), vocab_(vocab) {}

InferenceEngine::~InferenceEngine() {
  if (sampler_) llama_sampler_free(sampler_);
  if (ctx_) llama_free(ctx_);
  if (model_) llama_model_free(model_);
}

absl::StatusOr<std::unique_ptr<InferenceEngine>> InferenceEngine::Create(
    const std::string& model_path) {
  // Register no-op logger to keep logs clean.
  llama_log_set(LlamaNoOpLogger, nullptr);

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = kGpuLayers;

  LOG(INFO) << "Loading LLM from " << model_path << "...";
  llama_model* model =
      llama_model_load_from_file(model_path.c_str(), model_params);
  if (!model) {
    return absl::InternalError(
        absl::StrCat("Failed to load model from ", model_path));
  }

  const llama_vocab* vocab = llama_model_get_vocab(model);
  return absl::WrapUnique(new InferenceEngine(model, vocab));
}

absl::StatusOr<std::string> InferenceEngine::Infer(const std::string& prompt) {
  absl::MutexLock lock(&mutex_);
  return InferInternal(prompt);
}

absl::StatusOr<std::string> InferenceEngine::InferInternal(
    const std::string& prompt) {
  // 1. Apply Gemma Chat Template
  std::string formatted_prompt = absl::StrCat(
      "<start_of_turn>user\n", prompt, "<end_of_turn>\n<start_of_turn>model\n");

  // 2. Tokenize formatted prompt
  std::vector<llama_token> tokens;
  int n_prompt = -llama_tokenize(vocab_, formatted_prompt.c_str(),
                                 formatted_prompt.size(), NULL, 0, true, true);
  tokens.resize(n_prompt);
  if (llama_tokenize(vocab_, formatted_prompt.c_str(), formatted_prompt.size(),
                     tokens.data(), tokens.size(), true, true) < 0) {
    return absl::InternalError("Failed to tokenize prompt");
  }

  // 3. Re-initialize Context per request (matching reference behavior for
  // safety)
  if (ctx_) llama_free(ctx_);

  llama_context_params ctx_params = llama_context_default_params();
  // Ensure context is large enough for prompt + generation
  ctx_params.n_ctx = n_prompt + kMaxTokensToPredict;
  // Ensure batch size can handle the immediate prompt processing
  ctx_params.n_batch = n_prompt;
  ctx_params.no_perf = true;

  ctx_ = llama_init_from_model(model_, ctx_params);
  if (!ctx_) return absl::InternalError("Failed to create llama_context");

  // Initialize sampler if needed
  if (!sampler_) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    sampler_ = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());
  }

  // 4. Generation Loop
  std::string output;
  // Initial batch processing for the whole prompt
  llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

  int n_decode = 0;
  llama_token new_token_id;

  // Loop until we hit the token limit or EOG
  while (n_decode < kMaxTokensToPredict) {
    // Evaluate current batch
    if (llama_decode(ctx_, batch) != 0) {
      return absl::InternalError("llama_decode failed");
    }

    // Sample next token
    new_token_id = llama_sampler_sample(sampler_, ctx_, -1);

    // Check for End of Generation (EOG)
    if (llama_vocab_is_eog(vocab_, new_token_id)) {
      break;
    }

    // Convert token to text and append to output
    char buf[128];
    int n =
        llama_token_to_piece(vocab_, new_token_id, buf, sizeof(buf), 0, true);
    if (n < 0) return absl::InternalError("Failed to convert token to piece");
    output.append(buf, n);

    // Prepare batch for next single token
    batch = llama_batch_get_one(&new_token_id, 1);
    n_decode++;
  }

  return output;
}

}  // namespace gcp_prototype
