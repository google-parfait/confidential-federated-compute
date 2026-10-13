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

#include "containers/common/local_inference/inference_model.h"

#include <algorithm>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "common/common.h"  // From @llama_cpp.
#include "fcp/base/status_converters.h"
#include "gemma/gemma_args.h"
#include "include/llama-cpp.h"
#include "include/llama.h"

namespace confidential_federated_compute::local_inference {
namespace {

using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::RuntimeConfig;
using ::gcpp::Gemma;
using ::gcpp::InferenceArgs;
using ::gcpp::KVCache;
using ::gcpp::LoaderArgs;
using ::gcpp::MatMulEnv;
using ::gcpp::ThreadingArgs;
using ::gcpp::ThreadingContext;
using ::gcpp::TimingInfo;
using ::gcpp::WrapAndTokenize;

// Set the default max_output_tokens to 128. This can be overridden by the
// RuntimeConfig.max_generated_tokens field in the inference_configuration_.
constexpr size_t kMaxOutputTokens = 1024;
constexpr size_t kNumTokensPerBatch = 2048;
// Default maximum number of rows to process in a single GenerateBatch call.
constexpr size_t kDefaultMaxBatchSize = 1;

}  // namespace

InferenceModel::InferenceModel(
    SessionInferenceConfiguration inference_configuration,
    std::variant<std::shared_ptr<GemmaCppModel>, LlamaCppModel> model)
    : inference_configuration_(std::move(inference_configuration)),
      model_(std::move(model)) {}

absl::StatusOr<std::shared_ptr<GemmaCppModel>>
InferenceModel::BuildGemmaCppModel(
    const SessionInferenceConfiguration& inference_configuration) {
  if (!inference_configuration.gemma_configuration.has_value()) {
    return absl::InvalidArgumentError(
        "Missing session gemma.cpp configuration.");
  }
  const auto& gemma_config = *inference_configuration.gemma_configuration;
  auto gemma_model = std::make_shared<GemmaCppModel>();
  LoaderArgs loader_args(gemma_config.tokenizer_path,
                         gemma_config.model_weight_path);
  InferenceArgs inference_args;
  size_t seq_len =
      inference_configuration.initialize_configuration.inference_config()
          .runtime_config()
          .seq_len();
  if (seq_len > 0) {
    inference_args.seq_len = seq_len;
  }
  ThreadingArgs threading_args;
  ThreadingContext loading_ctx(threading_args);
  try {
    gemma_model->gemma =
        std::make_unique<Gemma>(loader_args, inference_args, loading_ctx);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to load gemma.cpp model: ", e.what()));
  }
  gemma_model->ctx = std::make_unique<ThreadingContext>(threading_args);
  gemma_model->env = std::make_unique<MatMulEnv>(*gemma_model->ctx);
  return gemma_model;
}

absl::StatusOr<LlamaCppModel> InferenceModel::BuildLlamaCppModel(
    const SessionLlamaCppConfiguration& llama_config) {
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = llama_config.n_gpu_layers;

  // Temporarily store the llama_model in a raw pointer.
  llama_model* raw_model = llama_model_load_from_file(
      llama_config.model_weight_path.c_str(), model_params);
  if (raw_model == NULL) {
    return absl::InternalError("Unable to load the llama.cpp model.");
  }
  LlamaCppModel llama_model;
  // Reset the unique_ptr to take ownership of the llama_model.
  llama_model.llama_.reset(raw_model);

  return llama_model;
}

absl::StatusOr<InferenceModel> InferenceModel::BuildModel(
    const SessionInferenceConfiguration& inference_configuration) {
  switch (inference_configuration.initialize_configuration
              .model_init_config_case()) {
    case InferenceInitializeConfiguration::kGemmaInitConfig: {
      // TODO: remove the ability to 'share' a GemmaCppModel across multiple
      // InferenceModel instances (which only fed_sql makes use of). This would
      // simplify the initialization code path further.
      if (inference_configuration.shared_gemma_model != nullptr) {
        if (inference_configuration.shared_gemma_model->gemma == nullptr) {
          return absl::InvalidArgumentError(
              "SessionInferenceConfiguration must contain a valid shared Gemma "
              "model.");
        }
        return InferenceModel(inference_configuration,
                              inference_configuration.shared_gemma_model);
      }
      ABSL_ASSIGN_OR_RETURN(std::shared_ptr<GemmaCppModel> gemma_model,
                            BuildGemmaCppModel(inference_configuration));
      return InferenceModel(inference_configuration, std::move(gemma_model));
    }
    case InferenceInitializeConfiguration::kLlamaCppInitConfig: {
      if (!inference_configuration.llama_configuration.has_value()) {
        return absl::InvalidArgumentError(
            "Missing session llama.cpp configuration.");
      }
      ABSL_ASSIGN_OR_RETURN(
          LlamaCppModel llama_model,
          BuildLlamaCppModel(*inference_configuration.llama_configuration));
      return InferenceModel(inference_configuration, std::move(llama_model));
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported model_init_config_case: ",
                       inference_configuration.initialize_configuration
                           .model_init_config_case()));
  }
}

std::vector<absl::StatusOr<std::string>> InferenceModel::GenerateBatchGemmaCpp(
    absl::Span<const std::string> prompts) {
  std::vector<absl::StatusOr<std::string>> results;
  results.reserve(prompts.size());

  GemmaCppModel& gemma_model =
      *std::get<std::shared_ptr<GemmaCppModel>>(model_);
  Gemma* gemma = gemma_model.gemma.get();

  RuntimeConfig inference_runtime_config =
      inference_configuration_.initialize_configuration.inference_config()
          .runtime_config();
  size_t config_max_output_tokens =
      inference_configuration_.initialize_configuration.inference_config()
          .runtime_config()
          .max_generated_tokens();
  size_t max_output_tokens = config_max_output_tokens > 0
                                 ? config_max_output_tokens
                                 : kMaxOutputTokens;
  float temperature = 1.0 + inference_runtime_config.temperature_diff();

  gcpp::MatMulEnv* env_ptr = gemma_model.env.get();
  gcpp::Allocator* allocator_ptr = &gemma_model.ctx->allocator;
  TimingInfo timing_info;

  size_t max_batch_size = inference_runtime_config.max_batch_size() > 0
                              ? inference_runtime_config.max_batch_size()
                              : kDefaultMaxBatchSize;

  // Process prompts in batches of max_batch_size (see GenerateBatch() in
  // inference_model.h).
  for (size_t batch_start = 0; batch_start < prompts.size();
       batch_start += max_batch_size) {
    const size_t batch_end =
        std::min(batch_start + max_batch_size, prompts.size());
    const size_t batch_size = batch_end - batch_start;

    // Tokenize all prompts in this batch.
    std::vector<std::vector<int>> all_tokens(batch_size);
    std::vector<gcpp::PromptTokens> prompt_tokens(batch_size);
    std::vector<size_t> prompt_sizes(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      std::string combined_prompt = prompts[batch_start + i];
      size_t generated = 0;
      all_tokens[i] =
          WrapAndTokenize(gemma->Tokenizer(), gemma->ChatTemplate(),
                          gemma->Config().wrapping, generated, combined_prompt);
      prompt_sizes[i] = all_tokens[i].size();
      prompt_tokens[i] =
          gcpp::PromptTokens(all_tokens[i].data(), all_tokens[i].size());
    }

    // Create per-query KV caches.
    std::vector<KVCache> kv_caches;
    kv_caches.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      kv_caches.push_back(
          KVCache(gemma->Config(), gemma->Inference(), *allocator_ptr));
    }

    hwy::Span<const gcpp::PromptTokens> prompt_span(prompt_tokens.data(),
                                                    prompt_tokens.size());
    hwy::Span<KVCache> kv_span(kv_caches.data(), kv_caches.size());
    gcpp::AllQueries all_queries(prompt_span, kv_span);

    // Set up per-query output streams and generated counters.
    std::vector<std::stringstream> output_streams(batch_size);
    std::vector<size_t> generated_counts(batch_size, 0);

    auto batch_stream_token = [&gemma, &output_streams, &generated_counts,
                               &prompt_sizes](size_t query_idx, size_t /*pos*/,
                                              int token,
                                              float /*prob*/) -> bool {
      // query_idx is absolute within AllQueries.
      size_t local_idx = query_idx;  // batch is rebuilt per chunk, so 0-based.
      ++generated_counts[local_idx];
      if (generated_counts[local_idx] >= prompt_sizes[local_idx] &&
          !gemma->Config().IsEOS(token)) {
        std::string token_text;
        if (!gemma->Tokenizer().Decode({token}, &token_text)) {
          LOG(WARNING) << "Failed to decode the next token.";
          return false;
        }
        output_streams[local_idx] << token_text;
      }
      return true;
    };

    gcpp::RuntimeConfig runtime_config = {
        .max_generated_tokens = max_output_tokens,
        .temperature = temperature,
        .verbosity = 0,
        .batch_stream_token = batch_stream_token,
    };
    // Serialize GenerateBatch calls on this model's thread pool / MatMulEnv.
    std::optional<std::string> chunk_error;
    {
      absl::MutexLock lock(&gemma_model.mutex);
      try {
        gemma->GenerateBatch(runtime_config, all_queries, *env_ptr,
                             timing_info);
      } catch (const std::exception& e) {
        chunk_error = e.what();
      }
    }
    if (chunk_error.has_value()) {
      // All prompts in a chunk share a single GenerateBatch call, so a failure
      // there invalidates the whole chunk. Other chunks are unaffected.
      for (size_t i = 0; i < batch_size; ++i) {
        results.push_back(absl::InternalError(absl::StrCat(
            "gemma.cpp batched inference failed: ", *chunk_error)));
      }
      continue;
    }

    // Collect outputs from stream callbacks.
    for (size_t i = 0; i < batch_size; ++i) {
      results.push_back(output_streams[i].str());
    }
  }

  return results;
}

absl::StatusOr<std::string> InferenceModel::RunLlamaCppInferencePerRow(
    const std::string& combined_prompt, LlamaCppModel& llama_model,
    const llama_vocab* vocab) {
  // 1. Tokenize the prompt.
  const int32_t num_prompt_tokens =
      -llama_tokenize(vocab, combined_prompt.c_str(), combined_prompt.size(),
                      /* tokens= */ NULL, /* n_tokens_max= */ 0,
                      /* add_special= */ true, /* parse_special= */ true);
  // Allocate space for the tokens and tokenize the prompt.
  std::vector<llama_token> prompt_tokens(num_prompt_tokens);
  // Actually populate the prompt tokens, and raise an error if the
  // tokenization fails.
  if (llama_tokenize(vocab, combined_prompt.c_str(), combined_prompt.size(),
                     prompt_tokens.data(), prompt_tokens.size(), true,
                     true) < 0) {
    return absl::InternalError("Failed to tokenize the prompt");
  }

  // 2. Initialize the context.
  llama_context_params ctx_params = llama_context_default_params();

  // Set the max_output_tokens to the value in the RuntimeConfig if provided.
  size_t config_max_output_tokens =
      inference_configuration_.initialize_configuration.inference_config()
          .runtime_config()
          .max_generated_tokens();
  size_t max_output_tokens = config_max_output_tokens > 0
                                 ? config_max_output_tokens
                                 : kMaxOutputTokens;
  // Set the context size to accommodate the prompt and generated tokens.
  ctx_params.n_ctx = num_prompt_tokens + max_output_tokens;
  // n_batch is the maximum number of tokens that can be processed in a single
  // call to llama_decode.
  ctx_params.n_batch = kNumTokensPerBatch;
  // Disable performance counters.
  ctx_params.no_perf = false;

  // Create the llama_context from the model, wrapped in a unique_ptr
  // (llama_context_ptr) to ensure automatic cleanup via llama_free.
  llama_context_ptr ctx(
      llama_init_from_model(llama_model.llama_.get(), ctx_params));
  if (!ctx) {
    return absl::InternalError("Failed to create the llama_context");
  }

  // 3. Initialize the sampler.
  auto sampler_params = llama_sampler_chain_default_params();
  sampler_params.no_perf = false;
  // Create the sampler chain, wrapped in a unique_ptr (llama_sampler_ptr) to
  // ensure automatic cleanup via llama_sampler_free.
  llama_sampler_ptr smpl(llama_sampler_chain_init(sampler_params));
  if (!smpl) {
    return absl::InternalError("Failed to create llama_sampler");
  }
  // Add a greedy sampler to the chain.
  llama_sampler_chain_add(smpl.get(), llama_sampler_init_greedy());

  // 4. Create and process the initial prompt batch
  // Initialize a llama_batch to hold the prompt tokens.
  struct llama_batch batch = llama_batch_init(
      /* n_tokens= */ num_prompt_tokens, /* embed= */ 0, /* n_seq_max= */ 1);
  // RAII wrapper to ensure llama_batch_free is called for the batch.
  auto batch_cleaner = [](struct llama_batch* b) { llama_batch_free(*b); };
  std::unique_ptr<struct llama_batch, decltype(batch_cleaner)> batch_ptr(
      &batch, batch_cleaner);

  // Sequence ID for the single stream
  const std::vector<llama_seq_id> seq_ids = {0};

  // Add all prompt tokens to the batch.
  for (int32_t i = 0; i < num_prompt_tokens; ++i) {
    // Add token at its position. Set request_logits to false for all prompt
    // tokens except the very last one.
    common_batch_add(/* batch= */ batch, /* id= */ prompt_tokens[i],
                     /* pos= */ i, /* seq_ids= */ seq_ids, /* logits= */ false);
  }
  // We need logits for the next token prediction.
  batch.logits[batch.n_tokens - 1] = true;

  // Process the entire prompt. This "ingests" the prompt and populates the KV
  // cache.
  if (llama_decode(ctx.get(), batch) != 0) {
    return absl::InternalError("Failed to decode prompt");
  }

  // 5. Loop for generating output tokens.
  std::stringstream output_stream;
  // n_cur tracks the current total sequence length.
  int32_t n_cur = batch.n_tokens;

  for (int i = 0; i < max_output_tokens; ++i) {
    // Sample the next token id.
    // sampler needs to know which token's logits to use, which is the last one
    // added.
    int32_t last_token_idx_in_batch = batch.n_tokens - 1;
    llama_token new_token_id =
        llama_sampler_sample(smpl.get(), ctx.get(), last_token_idx_in_batch);

    // Check if the sampled token is the End-of-Generation token.
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      break;
    }

    // Convert the sampled token ID back to its string representation.
    char buf[128];
    int32_t num_bytes_written =
        llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
    if (num_bytes_written < 0) {
      return absl::InternalError("Failed to convert token to piece.");
    }
    output_stream << std::string(buf, num_bytes_written);

    // Prepare the batch for the single new token.
    common_batch_clear(batch);
    // Add the new token at the current end of the sequence (n_cur).
    common_batch_add(batch, new_token_id, n_cur, seq_ids, true);

    // Increment the total sequence length.
    n_cur++;

    // Process the batch containing the single new token, which updates the KV
    // cache with the new token.
    if (llama_decode(ctx.get(), batch) != 0) {
      return absl::InternalError("Failed to decode next token");
    }
  }

  // Before returning, batch_ptr's destructor automatically calls
  // llama_batch_free(batch); ctx and smpl unique_ptrs handle their
  // cleanup.
  return output_stream.str();
}

std::vector<absl::StatusOr<std::string>> InferenceModel::GenerateBatchLlamaCpp(
    absl::Span<const std::string> prompts) {
  LlamaCppModel& llama_model = std::get<LlamaCppModel>(model_);
  const llama_vocab* vocab = llama_model_get_vocab(llama_model.llama_.get());
  if (!vocab) {
    return std::vector<absl::StatusOr<std::string>>(
        prompts.size(), absl::InternalError("Failed to get llama vocab"));
  }

  std::vector<absl::StatusOr<std::string>> results;
  results.reserve(prompts.size());
  // llama.cpp has no batched generation API here, so each prompt is run on its
  // own. Each element of the output corresponds to one input prompt.
  for (const std::string& prompt : prompts) {
    results.push_back(RunLlamaCppInferencePerRow(prompt, llama_model, vocab));
  }
  return results;
}

std::vector<absl::StatusOr<std::string>> InferenceModel::GenerateBatch(
    absl::Span<const std::string> prompts) {
  if (prompts.empty()) {
    return {};
  }
  if (std::holds_alternative<std::shared_ptr<GemmaCppModel>>(model_)) {
    return GenerateBatchGemmaCpp(prompts);
  }
  if (std::holds_alternative<LlamaCppModel>(model_)) {
    return GenerateBatchLlamaCpp(prompts);
  }
  return std::vector<absl::StatusOr<std::string>>(
      prompts.size(),
      absl::UnimplementedError("Unsupported inference model type."));
}

const SessionInferenceConfiguration& InferenceModel::GetInferenceConfiguration()
    const {
  return inference_configuration_;
}

std::shared_ptr<GemmaCppModel> InferenceModel::GetGemmaCppModel() const {
  if (const auto* gemma_model =
          std::get_if<std::shared_ptr<GemmaCppModel>>(&model_)) {
    return *gemma_model;
  }
  return nullptr;
}

}  // namespace confidential_federated_compute::local_inference
