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

#include "llama_cpp_batched_inference_engine.h"

#include <iostream>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "google/rpc/code.pb.h"

namespace confidential_federated_compute::gcp {
namespace {

// Hardcoded parameters for this prototype step.
constexpr int kDefaultMaxTokens = 1024;
constexpr int kBatchSize = 2048;  // Capacity for batch processing

// Custom log callback to suppress verbose output from llama.cpp
static void LlamaNoOpLogger(ggml_log_level level, const char* text,
                            void* user_data) {
  if (level == GGML_LOG_LEVEL_ERROR) {
    LOG(ERROR) << "llama.cpp: " << text;
  }
}

// Custom log callback to forward llama.cpp logs to Abseil
static void LlamaVerboseLogger(ggml_log_level level, const char* text,
                               void* user_data) {
  // Strip trailing newline for cleaner formatting
  std::string msg(text);
  if (!msg.empty() && msg.back() == '\n') {
    msg.pop_back();
  }

  if (level == GGML_LOG_LEVEL_ERROR) {
    LOG(ERROR) << "llama.cpp: " << msg;
  } else if (level == GGML_LOG_LEVEL_WARN) {
    LOG(WARNING) << "llama.cpp: " << msg;
  } else {
    // Capture INFO and DEBUG logs to verify GPU offloading stats
    LOG(INFO) << "llama.cpp: " << msg;
  }
}

// Chat template markers used to wrap prompts for instruction-tuned models.
// These are also used as stop strings during decoding: if the model emits
// them as regular text (which happens when the GGUF vocab doesn't register
// them as special tokens), we truncate the output at that point.
constexpr char kStartOfTurn[] = "<start_of_turn>";
constexpr char kEndOfTurn[] = "<end_of_turn>";

// --- Helper Functions for Manual Batch Manipulation ---

void BatchClear(llama_batch& batch) { batch.n_tokens = 0; }

absl::Status BatchAddSafe(llama_batch& batch, int capacity, llama_token token,
                          llama_pos pos,
                          const std::vector<llama_seq_id>& seq_ids,
                          bool logits) {
  if (batch.n_tokens >= capacity) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Batch full: ", batch.n_tokens, " >= ", capacity));
  }
  batch.token[batch.n_tokens] = token;
  batch.pos[batch.n_tokens] = pos;
  batch.n_seq_id[batch.n_tokens] = seq_ids.size();
  for (size_t i = 0; i < seq_ids.size(); ++i) {
    batch.seq_id[batch.n_tokens][i] = seq_ids[i];
  }
  batch.logits[batch.n_tokens] = logits ? 1 : 0;
  batch.n_tokens++;
  return absl::OkStatus();
}

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
class LlamaCppBatchedInferenceEngine : public BatchedInferenceEngine {
 public:
  LlamaCppBatchedInferenceEngine(llama_model* model, const llama_vocab* vocab);

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

  virtual ~LlamaCppBatchedInferenceEngine() override;

  /**
   * @brief Performs batched inference on a list of prompts.
   *
   * This method processes multiple prompts in parallel using llama.cpp's
   * batch decoding capabilities.
   *
   * @param request The proto containing the list of prompts and parameters.
   * @return A response proto with results for every prompt.
   */
  virtual absl::StatusOr<BatchedInferenceResponse> DoBatchedInference(
      const BatchedInferenceRequest& request) override;

 private:
  // Helper to tokenize a prompt (including chat template application).
  absl::StatusOr<std::vector<llama_token>> Tokenize(const std::string& prompt);

  llama_model* model_;        // Owned by this class.
  const llama_vocab* vocab_;  // Owned by model_.

  // Stop strings discovered from the model vocabulary at init time.
  // These are the text representations of control tokens (e.g. <end_of_turn>)
  // that should terminate generation if they appear in the output.
  std::vector<std::string> stop_strings_;
  // Maximum length of any stop string, used for tail-window matching.
  size_t max_stop_string_length_ = 0;

  absl::Mutex mutex_;
  // Context (KV cache) and sampler are reused across requests to save
  // allocation time, but are reset at the start of each DoBatchedInference()
  // call to ensure stateless request handling.
  llama_context* ctx_ ABSL_GUARDED_BY(mutex_) = nullptr;
  int current_n_ctx_ ABSL_GUARDED_BY(mutex_) = 0;
  int current_n_seq_max_ ABSL_GUARDED_BY(mutex_) = 0;
  int current_n_batch_ ABSL_GUARDED_BY(mutex_) = 0;
  llama_sampler* sampler_ ABSL_GUARDED_BY(mutex_) = nullptr;

  // Reusable batch structure for llama.cpp to avoid frequent allocations.
  llama_batch batch_ ABSL_GUARDED_BY(mutex_);
  int current_batch_capacity_ ABSL_GUARDED_BY(mutex_) = 0;
};

LlamaCppBatchedInferenceEngine::LlamaCppBatchedInferenceEngine(
    llama_model* model, const llama_vocab* vocab)
    : model_(model), vocab_(vocab) {
  // Initialize the batch structure once.
  batch_ = llama_batch_init(kBatchSize, 0, 1);
  current_batch_capacity_ = kBatchSize;

  // Build the stop string list from two sources:
  absl::flat_hash_set<std::string> seen;
  auto add_stop = [&](std::string s) {
    if (!s.empty() && seen.insert(s).second) {
      stop_strings_.push_back(std::move(s));
    }
  };

  // 1. Chat template markers.  Our Tokenize() wraps prompts with these.
  //    If the GGUF doesn't register them as special tokens, llama_tokenize
  //    splits them into regular text pieces, and the model may reproduce
  //    them verbatim in its output.  Catch that here.
  add_stop(kEndOfTurn);
  add_stop(kStartOfTurn);

  // 2. Model vocab control tokens (belt-and-suspenders for models that
  //    DO encode their turn markers as native control tokens).
  auto maybe_add_token = [&](llama_token id) {
    if (id < 0) return;
    char buf[256];
    int n = llama_token_to_piece(vocab_, id, buf, sizeof(buf), 0, true);
    if (n > 0) add_stop(std::string(buf, n));
  };
  maybe_add_token(llama_vocab_eot(vocab_));
  int n_vocab = llama_vocab_n_tokens(vocab_);
  for (int i = 0; i < n_vocab; i++) {
    if (llama_vocab_get_attr(vocab_, i) & LLAMA_TOKEN_ATTR_CONTROL) {
      maybe_add_token(i);
    }
  }

  LOG(INFO) << "Stop strings (" << stop_strings_.size() << "):";
  for (const auto& s : stop_strings_) {
    LOG(INFO) << "  \"" << s << "\"";
    max_stop_string_length_ = std::max(max_stop_string_length_, s.size());
  }

  // Initialize sampler once (greedy). Reused across all DoBatchedInference
  // calls via llama_sampler_reset() instead of free/reinit per batch.
  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = true;
  sampler_ = llama_sampler_chain_init(sparams);
  llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());
}

LlamaCppBatchedInferenceEngine::~LlamaCppBatchedInferenceEngine() {
  if (sampler_) llama_sampler_free(sampler_);
  if (ctx_) llama_free(ctx_);
  llama_batch_free(batch_);
  if (model_) llama_model_free(model_);
}

absl::StatusOr<std::vector<llama_token>>
LlamaCppBatchedInferenceEngine::Tokenize(const std::string& prompt) {
  // Apply chat template using the shared turn marker constants.
  std::string formatted_prompt =
      absl::StrCat(kStartOfTurn, "user\n", prompt, kEndOfTurn, "\n",
                   kStartOfTurn, "model\n");

  // 2. Tokenize formatted prompt
  std::vector<llama_token> tokens;
  int n_prompt = -llama_tokenize(vocab_, formatted_prompt.c_str(),
                                 formatted_prompt.size(), NULL, 0, true, true);
  tokens.resize(n_prompt);
  if (llama_tokenize(vocab_, formatted_prompt.c_str(), formatted_prompt.size(),
                     tokens.data(), tokens.size(), true, true) < 0) {
    return absl::InvalidArgumentError("Failed to tokenize prompt");
  }
  return tokens;
}

absl::StatusOr<BatchedInferenceResponse>
LlamaCppBatchedInferenceEngine::DoBatchedInference(
    const BatchedInferenceRequest& request) {
  absl::MutexLock lock(&mutex_);

  BatchedInferenceResponse response;
  if (request.requests().empty()) {
    return response;
  }

  // 1. Per-prompt tokenization with fault isolation.
  struct PromptState {
    int seq_id;
    std::vector<llama_token> tokens;
    std::string output;
    bool done;
    int tokens_generated;
    google::rpc::Code error_code;
    std::string error_message;
  };

  std::vector<PromptState> states;
  states.reserve(request.requests_size());

  for (int i = 0; i < request.requests_size(); ++i) {
    PromptState state;
    state.seq_id = i;
    state.done = false;
    state.tokens_generated = 0;
    state.error_code = google::rpc::Code::OK;

    // Prefer `prompt` (bytes, UTF-8-safe); fall back to deprecated `text`.
    const auto& req = request.requests(i);
    std::string prompt_content;
    if (!req.prompt().empty() && !req.text().empty()) {
      state.done = true;
      state.error_code = google::rpc::Code::INVALID_ARGUMENT;
      state.error_message =
          "Both 'prompt' and 'text' set on InferenceRequest; use only "
          "'prompt'.";
    } else if (!req.prompt().empty()) {
      prompt_content = std::string(req.prompt());
    } else {
      prompt_content = req.text();
    }

    if (!state.done) {
      auto tokens_or = Tokenize(prompt_content);
      if (!tokens_or.ok()) {
        state.done = true;
        state.error_code = google::rpc::Code::INVALID_ARGUMENT;
        state.error_message = std::string(tokens_or.status().message());
      } else {
        state.tokens = std::move(*tokens_or);
      }
    }
    states.push_back(std::move(state));
  }

  int max_tokens_to_predict = request.params().max_output_tokens() > 0
                                  ? request.params().max_output_tokens()
                                  : kDefaultMaxTokens;

  // Compute active count and prefill tokens (valid prompts only).
  int active_count = 0;
  int prefill_tokens = 0;
  for (const auto& s : states) {
    if (!s.done) {
      active_count++;
      prefill_tokens += s.tokens.size();
    }
  }

  // Fast-path: if all prompts failed tokenization, skip straight to
  // response assembly — no inference needed.
  if (active_count == 0) {
    for (const auto& state : states) {
      auto* result = response.add_results();
      result->set_text("");
      result->mutable_status()->set_code(state.error_code);
      result->mutable_status()->set_message(state.error_message);
    }
    return response;
  }

  // total_tokens_needed includes room for generation (active prompts only).
  int total_tokens_needed =
      prefill_tokens + (active_count * max_tokens_to_predict);

  // Capacity for the batch struct itself only needs
  // to hold the prefill pass (decode step adds 1 per
  // sequence, which is always <= prefill_tokens).
  int capacity = std::max(static_cast<int>(kBatchSize), prefill_tokens + 128);

  // Reallocate batch if current allocation is too small.
  if (capacity > current_batch_capacity_) {
    llama_batch new_batch = llama_batch_init(capacity, 0, 1);
    if (new_batch.token == nullptr || new_batch.pos == nullptr ||
        new_batch.n_seq_id == nullptr || new_batch.seq_id == nullptr ||
        new_batch.logits == nullptr) {
      llama_batch_free(new_batch);
      return absl::ResourceExhaustedError("Failed to allocate batch buffer");
    }
    llama_batch_free(batch_);
    batch_ = new_batch;
    current_batch_capacity_ = capacity;
  }

  // Reuse context if existing allocation is large enough; otherwise
  // reallocate.  llama_memory_clear() resets the KV cache head position,
  // making all prior batch state unreachable by the decode loop — safe for
  // TEE privacy (no cross-batch information leakage).
  int needed_n_ctx = total_tokens_needed + 128;
  int needed_n_seq_max = request.requests_size();
  if (ctx_ && needed_n_ctx <= current_n_ctx_ &&
      needed_n_seq_max <= current_n_seq_max_ && capacity <= current_n_batch_) {
    llama_memory_clear(llama_get_memory(ctx_), true);
  } else {
    if (ctx_) llama_free(ctx_);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = needed_n_ctx;
    ctx_params.n_batch = capacity;
    ctx_params.no_perf = true;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_params.n_seq_max = needed_n_seq_max;
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
      return absl::InternalError("Failed to create llama_context");
    }
    current_n_ctx_ = needed_n_ctx;
    current_n_seq_max_ = needed_n_seq_max;
    current_n_batch_ = capacity;
  }

  // Reset the persistent sampler for the new batch.
  llama_sampler_reset(sampler_);

  // 4. Batch Generation Loop
  BatchClear(batch_);

  // Initial Prefill — skip failed prompts.
  for (auto& state : states) {
    if (state.done) continue;
    for (size_t k = 0; k < state.tokens.size(); ++k) {
      auto s = BatchAddSafe(batch_, capacity, state.tokens[k], k,
                            {state.seq_id}, k == state.tokens.size() - 1);
      if (!s.ok()) {
        state.done = true;
        state.error_code = google::rpc::Code::RESOURCE_EXHAUSTED;
        state.error_message = std::string(s.message());
        active_count--;
        break;
      }
    }
  }

  int active_sequences = active_count;

  while (active_sequences > 0) {
    // llama_decode failure is a GPU/KV-cache level crash.
    // This is an accepted fatal error that will abort the
    // pipeline. Do NOT reclassify.
    if (llama_decode(ctx_, batch_) != 0) {
      return absl::InternalError("llama_decode failed");
    }

    // We need to queue up the NEXT tokens.
    // llama.cpp's sampler needs to know which index in the batch corresponds to
    // which sequence.
    std::vector<std::pair<int, llama_token>> next_step_inputs;

    for (int i = 0; i < batch_.n_tokens; ++i) {
      if (!batch_.logits[i]) continue;

      // batch_.seq_id[i][0] holds the sequence ID we assigned.
      int seq_id = batch_.seq_id[i][0];

      // Sample next token for this sequence
      llama_token new_token_id = llama_sampler_sample(sampler_, ctx_, i);

      // Check End of Generation conditions
      if (llama_vocab_is_eog(vocab_, new_token_id) ||
          states[seq_id].tokens_generated >= max_tokens_to_predict) {
        states[seq_id].done = true;
        active_sequences--;
      } else {
        size_t old_size = states[seq_id].output.size();
        // Convert to text
        char buf[128];
        // special=false: suppress tokens with LLAMA_TOKEN_ATTR_CONTROL.
        int n = llama_token_to_piece(vocab_, new_token_id, buf, sizeof(buf), 0,
                                     false);
        if (n < 0) {
          // Buffer too small. Retry with exact size.
          std::string large_buf(-n, '\0');
          int m = llama_token_to_piece(vocab_, new_token_id, large_buf.data(),
                                       large_buf.size(), 0, false);
          if (m > 0) {
            states[seq_id].output.append(large_buf.data(), m);
          }
        } else if (n > 0) {
          states[seq_id].output.append(buf, n);
        }
        states[seq_id].tokens_generated++;

        // Stop-string detection: some models emit turn markers as regular
        // tokens that aren't flagged as EOG in the GGUF vocab, so
        // llama_vocab_is_eog misses them.  Check against the stop strings
        // we discovered from the model vocabulary at init time.
        // Optimization: only search the tail window of the output buffer
        // anchored before the newly appended token instead of the full string.
        bool hit_stop = false;
        const auto& output = states[seq_id].output;
        size_t window_start = old_size > max_stop_string_length_
                                  ? old_size - max_stop_string_length_
                                  : 0;
        for (const auto& stop : stop_strings_) {
          auto pos = output.find(stop, window_start);
          if (pos != std::string::npos) {
            states[seq_id].output.resize(pos);
            hit_stop = true;
            break;
          }
        }

        if (hit_stop) {
          states[seq_id].done = true;
          active_sequences--;
        } else {
          // Prepare for next iteration
          next_step_inputs.push_back({seq_id, new_token_id});
        }
      }
    }

    // Reset batch for the next decoding step
    BatchClear(batch_);

    for (const auto& input : next_step_inputs) {
      int seq_id = input.first;
      llama_token token = input.second;
      // Position is length of prompt + generated so far
      int pos =
          states[seq_id].tokens.size() + states[seq_id].tokens_generated - 1;

      auto s = BatchAddSafe(batch_, capacity, token, pos, {seq_id}, true);
      if (!s.ok()) {
        LOG(WARNING) << "BatchAddSafe failed mid-decode for seq " << seq_id
                     << ": " << s;
        states[seq_id].done = true;
        states[seq_id].error_code = google::rpc::Code::RESOURCE_EXHAUSTED;
        states[seq_id].error_message = std::string(s.message());
        active_sequences--;
        continue;
      }
    }
  }

  // 5. Populate Response with per-item statuses.
  for (const auto& state : states) {
    auto* result = response.add_results();
    result->mutable_status()->set_code(state.error_code);
    if (state.error_code == google::rpc::Code::OK) {
      // Trim trailing whitespace/newlines.
      std::string output = state.output;
      while (!output.empty() &&
             (output.back() == '\n' || output.back() == ' ')) {
        output.pop_back();
      }
      result->set_response_text(output);
      // Also set deprecated `text` for backward compatibility with old clients.
      result->set_text(output);
    } else {
      result->set_response_text("");
      result->set_text("");
      result->mutable_status()->set_message(state.error_message);
    }
  }

  return response;
}

}  // namespace

absl::StatusOr<std::unique_ptr<BatchedInferenceEngine>>
CreateLlamaCppBatchedInferenceEngine(const std::string& model_path,
                                     int gpu_layers) {
  llama_log_set(LlamaVerboseLogger, nullptr);
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = gpu_layers;

  LOG(INFO) << "Loading LLM from " << model_path;
  LOG(INFO) << "Attempting to offload " << gpu_layers << " layers to GPU.";
  llama_model* model =
      llama_model_load_from_file(model_path.c_str(), model_params);
  if (!model) {
    return absl::InternalError(
        absl::StrCat("Failed to load model from ", model_path));
  }

  const llama_vocab* vocab = llama_model_get_vocab(model);
  return absl::WrapUnique(new LlamaCppBatchedInferenceEngine(model, vocab));
}

}  // namespace confidential_federated_compute::gcp
