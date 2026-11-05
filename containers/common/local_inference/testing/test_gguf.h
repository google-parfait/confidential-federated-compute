// Copyright 2026 Google LLC.
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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_LOCAL_INFERENCE_TESTING_TEST_GGUF_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_LOCAL_INFERENCE_TESTING_TEST_GGUF_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ggml.h"
#include "gguf.h"

namespace confidential_federated_compute::local_inference::testing {

// Writes a minimal synthetic GGUF v3 model file to `path` using `architecture`
// as both `general.architecture` and the hyperparameter key prefix.
//
// When `architecture` is `"gemma4"`, the file satisfies all Gemma 4
// hyperparameter, BPE tokenizer (`tokenizer.ggml.model = "gemma4"`), and tensor
// shape invariants required by `llama_model_load_from_file` in `llama.cpp`
// `b10400` and newer, while failing on older `llama.cpp` revisions that do not
// register `LLM_ARCH_GEMMA4`. Passing an unrecognized `architecture` string
// writes an identical GGUF structure that triggers the unknown-architecture
// error path during model loading.
inline bool WriteMinimalGemma4Gguf(absl::string_view path,
                                   absl::string_view architecture = "gemma4") {
  // Allocate enough backing memory for 2 tiny transformer layers of F32
  // weights (~8 KB of tensor data plus ggml_tensor struct overhead).
  constexpr size_t kGgmlPoolBytes = 512 * 1024;
  constexpr uint32_t kBlockCount = 2;
  constexpr uint32_t kContextLength = 64;
  constexpr uint32_t kEmbeddingLength = 8;
  constexpr uint32_t kFeedForwardLength = 16;
  constexpr uint32_t kHeadCount = 2;
  constexpr uint32_t kHeadCountKv = 1;
  constexpr uint32_t kHeadDim = 4;
  constexpr uint32_t kSlidingWindow = 16;
  constexpr float kLayerNormRmsEps = 1e-6f;
  constexpr size_t kVocabSize = 6;
  constexpr float kProjectionInitValue = 0.01f;
  constexpr float kNormInitValue = 1.0f;

  ggml_init_params init_params = {
      .mem_size = kGgmlPoolBytes,
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  std::unique_ptr<ggml_context, decltype(&ggml_free)> ggml_ctx(
      ggml_init(init_params), &ggml_free);
  if (ggml_ctx == nullptr) {
    return false;
  }

  std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf_ctx(
      gguf_init_empty(), &gguf_free);
  if (gguf_ctx == nullptr) {
    return false;
  }

  auto arch_key = [architecture](absl::string_view suffix) {
    return absl::StrCat(architecture, ".", suffix);
  };

  const std::string arch_str(architecture);
  gguf_set_val_str(gguf_ctx.get(), "general.architecture", arch_str.c_str());
  gguf_set_val_str(gguf_ctx.get(), "general.name", "synthetic-gemma4-test");
  gguf_set_val_u32(gguf_ctx.get(), arch_key("block_count").c_str(),
                   kBlockCount);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("context_length").c_str(),
                   kContextLength);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("embedding_length").c_str(),
                   kEmbeddingLength);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("feed_forward_length").c_str(),
                   kFeedForwardLength);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("attention.head_count").c_str(),
                   kHeadCount);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("attention.head_count_kv").c_str(),
                   kHeadCountKv);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("attention.key_length").c_str(),
                   kHeadDim);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("attention.value_length").c_str(),
                   kHeadDim);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("attention.key_length_swa").c_str(),
                   kHeadDim);
  gguf_set_val_u32(gguf_ctx.get(),
                   arch_key("attention.value_length_swa").c_str(), kHeadDim);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("rope.dimension_count").c_str(),
                   kHeadDim);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("rope.dimension_count_swa").c_str(),
                   kHeadDim);
  gguf_set_val_u32(gguf_ctx.get(), arch_key("attention.sliding_window").c_str(),
                   kSlidingWindow);
  // Layer 0 uses sliding-window attention and layer 1 uses full attention with
  // proportional RoPE frequency factors. GGUF stores bool arrays as int8 bytes.
  const int8_t kSlidingWindowPattern[kBlockCount] = {1, 0};
  gguf_set_arr_data(gguf_ctx.get(),
                    arch_key("attention.sliding_window_pattern").c_str(),
                    GGUF_TYPE_BOOL, kSlidingWindowPattern, kBlockCount);
  gguf_set_val_f32(gguf_ctx.get(),
                   arch_key("attention.layer_norm_rms_epsilon").c_str(),
                   kLayerNormRmsEps);
  gguf_set_val_u32(gguf_ctx.get(),
                   arch_key("embedding_length_per_layer_input").c_str(), 0);

  const char* kTokens[kVocabSize] = {"<pad>", "<eos>", "<bos>",
                                     "<unk>", "\n",    "hello"};
  const float kScores[kVocabSize] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  // Token types: 3 = control, 2 = unknown, 1 = normal.
  const int32_t kTokenTypes[kVocabSize] = {3, 3, 3, 2, 1, 1};
  const char* kMerges[1] = {"h e"};
  gguf_set_val_str(gguf_ctx.get(), "tokenizer.ggml.model", "gemma4");
  gguf_set_arr_str(gguf_ctx.get(), "tokenizer.ggml.tokens", kTokens,
                   kVocabSize);
  gguf_set_arr_data(gguf_ctx.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32,
                    kScores, kVocabSize);
  gguf_set_arr_data(gguf_ctx.get(), "tokenizer.ggml.token_type",
                    GGUF_TYPE_INT32, kTokenTypes, kVocabSize);
  gguf_set_arr_str(gguf_ctx.get(), "tokenizer.ggml.merges", kMerges, 1);
  gguf_set_val_u32(gguf_ctx.get(), "tokenizer.ggml.padding_token_id", 0);
  gguf_set_val_u32(gguf_ctx.get(), "tokenizer.ggml.eos_token_id", 1);
  gguf_set_val_u32(gguf_ctx.get(), "tokenizer.ggml.bos_token_id", 2);
  gguf_set_val_u32(gguf_ctx.get(), "tokenizer.ggml.unknown_token_id", 3);
  gguf_set_val_bool(gguf_ctx.get(), "tokenizer.ggml.add_bos_token", true);
  gguf_set_val_bool(gguf_ctx.get(), "tokenizer.ggml.add_space_prefix", false);

  auto add_1d = [&](absl::string_view name, int64_t ne0, float fill_value) {
    ggml_tensor* tensor =
        ggml_new_tensor_1d(ggml_ctx.get(), GGML_TYPE_F32, ne0);
    std::string tensor_name(name);
    ggml_set_name(tensor, tensor_name.c_str());
    std::fill_n(static_cast<float*>(tensor->data), ggml_nelements(tensor),
                fill_value);
    gguf_add_tensor(gguf_ctx.get(), tensor);
  };
  auto add_2d = [&](absl::string_view name, int64_t ne0, int64_t ne1,
                    float fill_value) {
    ggml_tensor* tensor =
        ggml_new_tensor_2d(ggml_ctx.get(), GGML_TYPE_F32, ne0, ne1);
    std::string tensor_name(name);
    ggml_set_name(tensor, tensor_name.c_str());
    std::fill_n(static_cast<float*>(tensor->data), ggml_nelements(tensor),
                fill_value);
    gguf_add_tensor(gguf_ctx.get(), tensor);
  };

  add_1d("output_norm.weight", kEmbeddingLength, kNormInitValue);
  add_1d("rope_freqs.weight", kHeadDim / 2, kNormInitValue);
  add_2d("token_embd.weight", kEmbeddingLength, kVocabSize,
         kProjectionInitValue);

  for (uint32_t layer = 0; layer < kBlockCount; ++layer) {
    add_1d(absl::StrCat("blk.", layer, ".attn_norm.weight"), kEmbeddingLength,
           kNormInitValue);
    add_2d(absl::StrCat("blk.", layer, ".attn_q.weight"), kEmbeddingLength,
           kHeadDim * kHeadCount, kProjectionInitValue);
    add_2d(absl::StrCat("blk.", layer, ".attn_k.weight"), kEmbeddingLength,
           kHeadDim * kHeadCountKv, kProjectionInitValue);
    add_2d(absl::StrCat("blk.", layer, ".attn_v.weight"), kEmbeddingLength,
           kHeadDim * kHeadCountKv, kProjectionInitValue);
    add_2d(absl::StrCat("blk.", layer, ".attn_output.weight"),
           kHeadDim * kHeadCount, kEmbeddingLength, kProjectionInitValue);
    add_1d(absl::StrCat("blk.", layer, ".attn_q_norm.weight"), kHeadDim,
           kNormInitValue);
    add_1d(absl::StrCat("blk.", layer, ".attn_k_norm.weight"), kHeadDim,
           kNormInitValue);
    add_1d(absl::StrCat("blk.", layer, ".post_attention_norm.weight"),
           kEmbeddingLength, kNormInitValue);
    add_1d(absl::StrCat("blk.", layer, ".ffn_norm.weight"), kEmbeddingLength,
           kNormInitValue);
    add_2d(absl::StrCat("blk.", layer, ".ffn_gate.weight"), kEmbeddingLength,
           kFeedForwardLength, kProjectionInitValue);
    add_2d(absl::StrCat("blk.", layer, ".ffn_up.weight"), kEmbeddingLength,
           kFeedForwardLength, kProjectionInitValue);
    add_2d(absl::StrCat("blk.", layer, ".ffn_down.weight"), kFeedForwardLength,
           kEmbeddingLength, kProjectionInitValue);
    add_1d(absl::StrCat("blk.", layer, ".post_ffw_norm.weight"),
           kEmbeddingLength, kNormInitValue);
  }

  const std::string output_path(path);
  return gguf_write_to_file(gguf_ctx.get(), output_path.c_str(),
                            /*only_meta=*/false);
}

}  // namespace confidential_federated_compute::local_inference::testing

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_LOCAL_INFERENCE_TESTING_TEST_GGUF_H_
