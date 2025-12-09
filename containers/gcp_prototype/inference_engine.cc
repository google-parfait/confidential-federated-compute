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
#include "google/rpc/code.pb.h"

namespace gcp_prototype {

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

// --- Helper Functions for Manual Batch Manipulation ---

void BatchClear(llama_batch& batch) { batch.n_tokens = 0; }

void BatchAdd(llama_batch& batch, llama_token token, llama_pos pos,
              const std::vector<llama_seq_id>& seq_ids, bool logits) {
  batch.token[batch.n_tokens] = token;
  batch.pos[batch.n_tokens] = pos;
  batch.n_seq_id[batch.n_tokens] = seq_ids.size();
  for (size_t i = 0; i < seq_ids.size(); ++i) {
    batch.seq_id[batch.n_tokens][i] = seq_ids[i];
  }
  batch.logits[batch.n_tokens] = logits ? 1 : 0;
  batch.n_tokens++;
}

}  // namespace

InferenceEngine::InferenceEngine(llama_model* model, const llama_vocab* vocab)
    : model_(model), vocab_(vocab) {
  // Initialize the batch structure once.
  // We allocate enough space for kBatchSize tokens.
  // The '1' determines the max number of sequences a single token can belong
  // to. Since we map 1 token to 1 sequence, 1 is sufficient.
  batch_ = llama_batch_init(kBatchSize, 0, 1);
}

InferenceEngine::~InferenceEngine() {
  if (sampler_) llama_sampler_free(sampler_);
  if (ctx_) llama_free(ctx_);
  llama_batch_free(batch_);
  if (model_) llama_model_free(model_);
}

absl::StatusOr<std::unique_ptr<InferenceEngine>> InferenceEngine::Create(
    const std::string& model_path, int gpu_layers) {
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
  return absl::WrapUnique(new InferenceEngine(model, vocab));
}

absl::StatusOr<std::string> InferenceEngine::Infer(const std::string& prompt) {
  // No lock needed here; we are just constructing a request on the stack.
  // The shared state access happens inside InferBatch.
  BatchedInferenceRequest request;
  request.add_requests()->set_text(prompt);
  request.mutable_params()->set_max_output_tokens(kDefaultMaxTokens);

  auto result_or = InferBatch(request);

  if (!result_or.ok()) {
    return result_or.status();
  }

  if (result_or->results_size() > 0) {
    const auto& result = result_or->results(0);
    if (result.status().code() == google::rpc::Code::OK) {
      return result.text();
    }
    return absl::InternalError(
        absl::StrCat("Inference failed: ", result.status().message()));
  }
  return absl::InternalError("No result returned from batch inference");
}

absl::StatusOr<std::vector<llama_token>> InferenceEngine::Tokenize(
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
  return tokens;
}

absl::StatusOr<BatchedInferenceResponse> InferenceEngine::InferBatch(
    const BatchedInferenceRequest& request) {
  absl::MutexLock lock(&mutex_);

  BatchedInferenceResponse response;
  if (request.requests().empty()) {
    return response;
  }

  // 1. Prepare inputs
  std::vector<std::vector<llama_token>> all_tokens;
  int total_tokens_needed = 0;

  for (const auto& req : request.requests()) {
    auto tokens_or = Tokenize(req.text());
    if (!tokens_or.ok()) return tokens_or.status();
    all_tokens.push_back(*tokens_or);
    total_tokens_needed += tokens_or->size();
  }

  int max_tokens_to_predict = request.params().max_output_tokens() > 0
                                  ? request.params().max_output_tokens()
                                  : kDefaultMaxTokens;

  // Add margin for generated tokens (batch size * max output)
  total_tokens_needed += (request.requests_size() * max_tokens_to_predict);

  // 2. Re-initialize Context per request
  if (ctx_) llama_free(ctx_);
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = total_tokens_needed + 128;  // Safety margin
  ctx_params.n_batch = kBatchSize;
  ctx_params.no_perf = true;
  // Allow enough unique sequences for every request in the batch
  ctx_params.n_seq_max = request.requests_size();

  ctx_ = llama_init_from_model(model_, ctx_params);
  if (!ctx_) return absl::InternalError("Failed to create llama_context");

  // 3. Initialize sampler
  if (sampler_) llama_sampler_free(sampler_);
  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = true;
  sampler_ = llama_sampler_chain_init(sparams);
  llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());

  // 4. Batch Generation Loop
  BatchClear(batch_);

  struct SeqState {
    int id;
    std::string output;
    bool done;
    int tokens_generated;
  };
  std::vector<SeqState> states;
  states.reserve(request.requests_size());

  // Initial Prefill
  for (int i = 0; i < request.requests_size(); ++i) {
    states.push_back({i, "", false, 0});
    const auto& tokens = all_tokens[i];
    for (size_t k = 0; k < tokens.size(); ++k) {
      // Request logits only for the last token of the prompt
      BatchAdd(batch_, tokens[k], k, {i}, k == tokens.size() - 1);
    }
  }

  int active_sequences = request.requests_size();

  while (active_sequences > 0) {
    // Evaluate current batch
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
        // Convert to text
        char buf[128];
        int n = llama_token_to_piece(vocab_, new_token_id, buf, sizeof(buf), 0,
                                     true);
        if (n > 0) {
          states[seq_id].output.append(buf, n);
        }
        states[seq_id].tokens_generated++;

        // Prepare for next iteration
        next_step_inputs.push_back({seq_id, new_token_id});
      }
    }

    // Reset batch for the next decoding step
    BatchClear(batch_);

    for (const auto& input : next_step_inputs) {
      int seq_id = input.first;
      llama_token token = input.second;
      // Position is length of prompt + generated so far
      int pos = all_tokens[seq_id].size() + states[seq_id].tokens_generated - 1;

      BatchAdd(batch_, token, pos, {seq_id}, true);
    }
  }

  // 5. Populate Response
  for (const auto& state : states) {
    auto* result = response.add_results();
    result->set_text(state.output);
    result->mutable_status()->set_code(google::rpc::Code::OK);
  }

  return response;
}

}  // namespace gcp_prototype
