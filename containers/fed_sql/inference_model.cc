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

#include "containers/fed_sql/inference_model.h"

#include <algorithm>
#include <regex>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "common/common.h"  // From @llama_cpp.
#include "containers/fed_sql/inference_model_helper.h"
#include "containers/sql/input.h"
#include "fcp/base/status_converters.h"
#include "gemma/gemma_args.h"
#include "include/llama-cpp.h"
#include "include/llama.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::confidential_federated_compute::sql::Input;
using ::confidential_federated_compute::sql::RowView;
using ::fcp::confidentialcompute::ColumnConfiguration;
using ::fcp::confidentialcompute::GEMMA2_2B;
using ::fcp::confidentialcompute::GEMMA2_9B;
using ::fcp::confidentialcompute::GEMMA3_12B;
using ::fcp::confidentialcompute::GEMMA3_1B;
using ::fcp::confidentialcompute::GEMMA3_4B;
using ::fcp::confidentialcompute::GEMMA_2B;
using ::fcp::confidentialcompute::GEMMA_7B;
using ::fcp::confidentialcompute::GEMMA_F32;
using ::fcp::confidentialcompute::GEMMA_IT;
using ::fcp::confidentialcompute::GEMMA_PT;
using ::fcp::confidentialcompute::GEMMA_SFP;
using ::fcp::confidentialcompute::GEMMA_TINY;
using ::fcp::confidentialcompute::GemmaConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::Prompt;
using ::fcp::confidentialcompute::RuntimeConfig;
using ::gcpp::Gemma;
using ::gcpp::InferenceArgs;
using ::gcpp::KVCache;
using ::gcpp::LoaderArgs;
using ::gcpp::MatMulEnv;
using ::gcpp::PromptWrapping;
using ::gcpp::ThreadingArgs;
using ::gcpp::ThreadingContext;
using ::gcpp::TimingInfo;
using ::gcpp::WrapAndTokenize;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorData;
using ::tensorflow_federated::aggregation::TensorShape;

// The max prompt size is set to match the gemma.cpp default seq_len value.
constexpr size_t kMaxPromptSize = 8192;
// Set the default max_output_tokens to 128. This can be overridden by the
// RuntimeConfig.max_generated_tokens field in the inference_configuration_.
constexpr size_t kMaxOutputTokens = 1024;
constexpr size_t kNumTokensPerBatch = 2048;

// Duplicates the data in a single column based on the per_row_output_counts.
// T is the C++ type of the data in the column.
template <typename T>
void DuplicateVectorData(const Tensor& original_column,
                         size_t original_row_count,
                         const std::vector<size_t>& per_row_output_counts,
                         TensorData* new_data_ptr) {
  auto* new_data = static_cast<MutableVectorData<T>*>(new_data_ptr);
  const auto original_span = original_column.AsSpan<T>();
  for (size_t i = 0; i < original_row_count; ++i) {
    const T& val = original_span[i];
    for (size_t k = 0; k < per_row_output_counts[i]; ++k) {
      new_data->push_back(val);
    }
  }
}

// Overload for DT_STRING.
void DuplicateStringData(const Tensor& original_column,
                         size_t original_row_count,
                         const std::vector<size_t>& per_row_output_counts,
                         TensorData* new_data_ptr) {
  auto* new_data = static_cast<MutableStringData*>(new_data_ptr);
  const auto original_span = original_column.AsSpan<absl::string_view>();
  for (size_t i = 0; i < original_row_count; ++i) {
    const absl::string_view val = original_span[i];
    for (size_t k = 0; k < per_row_output_counts[i]; ++k) {
      new_data->Add(std::string(val));
    }
  }
}

}  // namespace

absl::Status InferenceModel::BuildGemmaCppModel(
    const SessionGemmaCppConfiguration& gemma_config) {
  GemmaCppModel& gemma_model = std::get<GemmaCppModel>(model_);
  LoaderArgs loader_args(gemma_config.tokenizer_path,
                         gemma_config.model_weight_path);
  InferenceArgs inference_args;
  size_t seq_len =
      inference_configuration_->initialize_configuration.inference_config()
          .runtime_config()
          .seq_len();
  if (seq_len > 0) {
    inference_args.seq_len = seq_len;
  }
  ThreadingArgs threading_args;
  gemma_model.ctx_ = std::make_unique<ThreadingContext>(threading_args);
  gemma_model.gemma_ =
      std::make_unique<Gemma>(loader_args, inference_args, *gemma_model.ctx_);
  gemma_model.env_ = std::make_unique<MatMulEnv>(*gemma_model.ctx_);
  return absl::OkStatus();
}

absl::Status InferenceModel::BuildLlamaCppModel(
    const SessionLlamaCppConfiguration& llama_config) {
  LlamaCppModel& llama_model_ref = std::get<LlamaCppModel>(model_);

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = llama_config.n_gpu_layers;

  // Temporarily store the llama_model in a raw pointer.
  llama_model* raw_model = llama_model_load_from_file(
      llama_config.model_weight_path.c_str(), model_params);
  if (raw_model == NULL) {
    return absl::InternalError("Unable to load the llama.cpp model.");
  }
  // Reset the unique_ptr to take ownership of the llama_model.
  llama_model_ref.llama_.reset(raw_model);

  return absl::OkStatus();
}

absl::Status InferenceModel::BuildModel(
    const SessionInferenceConfiguration& inference_configuration) {
  inference_configuration_ = inference_configuration;
  switch (inference_configuration.initialize_configuration
              .model_init_config_case()) {
    case InferenceInitializeConfiguration::kGemmaInitConfig: {
      if (!inference_configuration.gemma_configuration.has_value()) {
        return absl::InvalidArgumentError(
            "Missing session gemma.cpp configuration.");
      }
      model_.emplace<GemmaCppModel>();
      FCP_RETURN_IF_ERROR(BuildGemmaCppModel(
          inference_configuration.gemma_configuration.value()));
      break;
    }
    case InferenceInitializeConfiguration::kLlamaCppInitConfig: {
      if (!inference_configuration.llama_configuration.has_value()) {
        return absl::InvalidArgumentError(
            "Missing session llama.cpp configuration.");
      }
      model_.emplace<LlamaCppModel>();
      FCP_RETURN_IF_ERROR(BuildLlamaCppModel(
          inference_configuration.llama_configuration.value()));
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported model_init_config_case: ",
                       inference_configuration.initialize_configuration
                           .model_init_config_case()));
      break;
  }
  return absl::OkStatus();
}

absl::StatusOr<InferenceModel::InferenceOutput>
InferenceModel::RunGemmaCppInference(
    const Prompt& prompt, const Input& input,
    absl::Span<const size_t> input_column_indices,
    const std::string& output_column_name) {
  if (input_column_indices.empty()) {
    FCP_ASSIGN_OR_RETURN(Tensor tensor,
                         Tensor::Create(DataType::DT_STRING, TensorShape({0}),
                                        std::make_unique<MutableStringData>(0),
                                        output_column_name));
    return InferenceOutput{std::move(tensor), std::vector<size_t>()};
  }
  // Only need to initialize this once.
  GemmaCppModel& gemma_model = std::get<GemmaCppModel>(model_);
  Gemma* gemma = gemma_model.gemma_.get();
  KVCache kv_cache(gemma->Config(), gemma->Inference(),
                   gemma_model.ctx_->allocator);
  RuntimeConfig inference_runtime_config =
      inference_configuration_->initialize_configuration.inference_config()
          .runtime_config();
  size_t max_prompt_size = inference_runtime_config.max_prompt_size() > 0
                               ? inference_runtime_config.max_prompt_size()
                               : kMaxPromptSize;
  std::unique_ptr<MutableStringData> output_string_data =
      std::make_unique<MutableStringData>(
          static_cast<long>(input.GetRowCount()));
  size_t num_output_rows = 0;
  std::vector<size_t> per_row_output_counts;
  per_row_output_counts.reserve(input.GetRowCount());

  TimingInfo timing_info;

  // For each row of the input columns, we first populate the prompt template
  // to create a combined prompt matching the column values, then run inference
  // over the combined prompt. Each element in the output tensor corresponds
  // to the inference result of one row of the input columns.
  for (int i = 0; i < input.GetRowCount(); ++i) {
    FCP_ASSIGN_OR_RETURN(RowView row, input.GetRow(i));
    FCP_ASSIGN_OR_RETURN(
        std::string combined_prompt,
        prompt_processor_.PopulatePromptTemplate(
            prompt, row, input.GetColumnNames(), input_column_indices,
            output_column_name, max_prompt_size));
    size_t generated = 0;
    const std::vector<int> tokens =
        WrapAndTokenize(gemma->Tokenizer(), gemma->ChatTemplate(),
                        gemma->Config().wrapping, generated, combined_prompt);
    const size_t prompt_size = tokens.size();
    std::stringstream output_stream;
    auto stream_token = [&gemma, &output_stream, &generated, &prompt_size](
                            int token, float) {
      ++generated;
      if (generated >= prompt_size && !gemma->Config().IsEOS(token)) {
        std::string token_text;
        if (!gemma->Tokenizer().Decode({token}, &token_text)) {
          LOG(WARNING) << "Failed to decode the next token.";
          return false;
        }
        output_stream << token_text;
      }
      return true;
    };
    float temperature = 1.0 + inference_runtime_config.temperature_diff();

    // Set the max_output_tokens to the value in the RuntimeConfig if provided.
    size_t config_max_output_tokens =
        inference_configuration_->initialize_configuration.inference_config()
            .runtime_config()
            .max_generated_tokens();
    size_t max_output_tokens = config_max_output_tokens > 0
                                   ? config_max_output_tokens
                                   : kMaxOutputTokens;
    gcpp::RuntimeConfig runtime_config = {
        .max_generated_tokens = max_output_tokens,
        .temperature = temperature,
        .gen = &gemma_model.gen_,
        .verbosity = 0,
        .stream_token = stream_token,
    };
    try {
      gemma->Generate(runtime_config, tokens, 0, kv_cache, *gemma_model.env_,
                      timing_info);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to run gemma.cpp inference";
    }

    std::string output_string = output_stream.str();
    absl::StatusOr<size_t> num_rows_added_status =
        output_processor_.ProcessInferenceOutput(
            prompt, std::move(output_string), output_column_name,
            output_string_data.get());
    if (!num_rows_added_status.ok()) {
      LOG(WARNING) << "Failed to process inference output: "
                   << num_rows_added_status.status()
                   << ". Outputting empty string.";
      output_string_data->Add("");
      num_output_rows++;
      per_row_output_counts.push_back(1);
    } else {
      num_output_rows += *num_rows_added_status;
      per_row_output_counts.push_back(*num_rows_added_status);
    }
  }

  FCP_ASSIGN_OR_RETURN(
      Tensor output_tensor,
      Tensor::Create(DataType::DT_STRING,
                     TensorShape({static_cast<long>(num_output_rows)}),
                     std::move(output_string_data), output_column_name));
  return InferenceOutput{std::move(output_tensor),
                         std::move(per_row_output_counts)};
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
      inference_configuration_->initialize_configuration.inference_config()
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

absl::StatusOr<InferenceModel::InferenceOutput>
InferenceModel::RunLlamaCppInference(
    const Prompt& prompt, const Input& input,
    absl::Span<const size_t> input_column_indices,
    const std::string& output_column_name) {
  if (input_column_indices.empty()) {
    FCP_ASSIGN_OR_RETURN(Tensor tensor,
                         Tensor::Create(DataType::DT_STRING, TensorShape({0}),
                                        std::make_unique<MutableStringData>(0),
                                        output_column_name));
    return InferenceOutput{std::move(tensor), std::vector<size_t>()};
  }
  LlamaCppModel& llama_model = std::get<LlamaCppModel>(model_);
  const llama_vocab* vocab = llama_model_get_vocab(llama_model.llama_.get());
  if (!vocab) {
    return absl::InternalError("Failed to get llama vocab");
  }
  RuntimeConfig inference_runtime_config =
      inference_configuration_->initialize_configuration.inference_config()
          .runtime_config();
  size_t max_prompt_size = inference_runtime_config.max_prompt_size() > 0
                               ? inference_runtime_config.max_prompt_size()
                               : kMaxPromptSize;

  std::unique_ptr<MutableStringData> output_string_data =
      std::make_unique<MutableStringData>(
          static_cast<long>(input.GetRowCount()));
  size_t num_output_rows = 0;
  std::vector<size_t> per_row_output_counts;
  per_row_output_counts.reserve(input.GetRowCount());
  // For each row of the input columns, we first populate the prompt template
  // to create a combined prompt matching the column values, then run inference
  // over the combined prompt. Each element in the output tensor corresponds
  // to the inference result of one row of the input columns.
  for (int i = 0; i < input.GetRowCount(); ++i) {
    FCP_ASSIGN_OR_RETURN(RowView row, input.GetRow(i));
    FCP_ASSIGN_OR_RETURN(
        std::string combined_prompt,
        prompt_processor_.PopulatePromptTemplate(
            prompt, row, input.GetColumnNames(), input_column_indices,
            output_column_name, max_prompt_size));
    // Generate inference output for a row.
    FCP_ASSIGN_OR_RETURN(
        std::string output_string,
        RunLlamaCppInferencePerRow(combined_prompt, llama_model, vocab));

    absl::StatusOr<size_t> num_rows_added_status =
        output_processor_.ProcessInferenceOutput(
            prompt, std::move(output_string), output_column_name,
            output_string_data.get());
    if (!num_rows_added_status.ok()) {
      LOG(WARNING) << "Failed to process inference output: "
                   << num_rows_added_status.status()
                   << ". Outputting empty string.";
      output_string_data->Add("");
      num_output_rows++;
      per_row_output_counts.push_back(1);
    } else {
      num_output_rows += *num_rows_added_status;
      per_row_output_counts.push_back(*num_rows_added_status);
    }
  }

  FCP_ASSIGN_OR_RETURN(
      Tensor output_tensor,
      Tensor::Create(DataType::DT_STRING,
                     TensorShape({static_cast<long>(num_output_rows)}),
                     std::move(output_string_data), output_column_name));
  return InferenceOutput{std::move(output_tensor),
                         std::move(per_row_output_counts)};
}

absl::Status InferenceModel::DuplicateColumnsForMultipleRows(
    sql::Input& input, std::map<std::string, Tensor>& output_columns,
    const std::map<std::string, std::vector<size_t>>&
        per_row_output_counts_map) {
  if (output_columns.size() != 1) {
    // This function is called per-task, so there is always one output column.
    return absl::InternalError(
        "DuplicateColumnsForMultipleRows called with multiple output columns");
  }
  const size_t original_row_count = input.GetRowCount();
  if (original_row_count == 0) {
    return absl::OkStatus();
  }

  FCP_ASSIGN_OR_RETURN(std::vector<Tensor> original_columns,
                       std::move(input).MoveToTensors());

  const auto& per_row_output_counts = per_row_output_counts_map.begin()->second;

  size_t total_new_rows = 0;
  for (size_t count : per_row_output_counts) {
    total_new_rows += count;
  }

  std::vector<std::unique_ptr<TensorData>> new_original_data;
  new_original_data.reserve(original_columns.size());
  for (const auto& col : original_columns) {
    switch (col.dtype()) {
      case DataType::DT_STRING: {
        auto new_data = std::make_unique<MutableStringData>(total_new_rows);
        DuplicateStringData(col, original_row_count, per_row_output_counts,
                            new_data.get());
        new_original_data.push_back(std::move(new_data));
        break;
      }
      case DataType::DT_INT64: {
        auto new_data = std::make_unique<MutableVectorData<int64_t>>();
        new_data->reserve(total_new_rows);
        DuplicateVectorData<int64_t>(col, original_row_count,
                                     per_row_output_counts, new_data.get());
        new_original_data.push_back(std::move(new_data));
        break;
      }
      case DataType::DT_INT32: {
        auto new_data = std::make_unique<MutableVectorData<int32_t>>();
        new_data->reserve(total_new_rows);
        DuplicateVectorData<int32_t>(col, original_row_count,
                                     per_row_output_counts, new_data.get());
        new_original_data.push_back(std::move(new_data));
        break;
      }
      case DataType::DT_FLOAT: {
        auto new_data = std::make_unique<MutableVectorData<float>>();
        new_data->reserve(total_new_rows);
        DuplicateVectorData<float>(col, original_row_count,
                                   per_row_output_counts, new_data.get());
        new_original_data.push_back(std::move(new_data));
        break;
      }
      case DataType::DT_DOUBLE: {
        auto new_data = std::make_unique<MutableVectorData<double>>();
        new_data->reserve(total_new_rows);
        DuplicateVectorData<double>(col, original_row_count,
                                    per_row_output_counts, new_data.get());
        new_original_data.push_back(std::move(new_data));
        break;
      }
      default:
        return absl::UnimplementedError(
            absl::StrCat("Unsupported data type for duplication: ",
                         DataType_Name(col.dtype())));
    }
  }

  std::vector<Tensor> final_columns;
  final_columns.reserve(original_columns.size());
  for (size_t i = 0; i < original_columns.size(); ++i) {
    FCP_ASSIGN_OR_RETURN(Tensor t,
                         Tensor::Create(original_columns[i].dtype(),
                                        TensorShape({(long)total_new_rows}),
                                        std::move(new_original_data[i]),
                                        original_columns[i].name()));
    final_columns.push_back(std::move(t));
  }
  FCP_ASSIGN_OR_RETURN(sql::Input new_input,
                       sql::Input::CreateFromTensors(std::move(final_columns),
                                                     {}));  // blob_header
  input = std::move(new_input);
  return absl::OkStatus();
}

absl::Status InferenceModel::RunInference(Input& input) {
  if (!HasModel()) {
    return absl::UnimplementedError(
        "Model must be initialized before running inference.");
  }

  for (const auto& inference_task :
       inference_configuration_->initialize_configuration.inference_config()
           .inference_task()) {
    if (!inference_task.has_prompt()) {
      return absl::InvalidArgumentError(
          "Prompt not found when running inference. Only prompt-based "
          "inference is supported.");
    }
    std::vector<std::string> input_column_names;
    if (!inference_task.column_config().input_column_names().empty()) {
      input_column_names.insert(
          input_column_names.end(),
          inference_task.column_config().input_column_names().begin(),
          inference_task.column_config().input_column_names().end());
    } else {
      return absl::InvalidArgumentError(
          "No input column names found when running inference.");
    }
    const std::string& output_column_name =
        inference_task.column_config().output_column_name();

    // Find the indices that correspond to the input columns.
    std::vector<size_t> input_column_indices;
    for (const auto& input_column_name : input_column_names) {
      const auto it =
          std::find(input.GetColumnNames().begin(),
                    input.GetColumnNames().end(), input_column_name);
      if (it == input.GetColumnNames().end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Couldn't find an input column ", input_column_name,
                         " to run inference on."));
      }
      input_column_indices.push_back(it - input.GetColumnNames().begin());
    }

    InferenceOutput inference_output;
    if (std::holds_alternative<GemmaCppModel>(model_)) {
      FCP_ASSIGN_OR_RETURN(
          inference_output,
          RunGemmaCppInference(inference_task.prompt(), input,
                               input_column_indices, output_column_name));
    } else if (std::holds_alternative<LlamaCppModel>(model_)) {
      FCP_ASSIGN_OR_RETURN(
          inference_output,
          RunLlamaCppInference(inference_task.prompt(), input,
                               input_column_indices, output_column_name));
    } else {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported inference model type."));
    }

    std::map<std::string, Tensor> current_output_column;
    current_output_column.emplace(output_column_name,
                                  std::move(inference_output.tensor));

    // If any of the rows generated multiple outputs, duplicate the columns to
    // match the new row count.
    if (std::any_of(inference_output.per_row_output_counts.begin(),
                    inference_output.per_row_output_counts.end(),
                    [](size_t count) { return count != 1; })) {
      std::map<std::string, std::vector<size_t>> per_row_output_counts_map;
      per_row_output_counts_map.emplace(
          output_column_name,
          std::move(inference_output.per_row_output_counts));

      FCP_RETURN_IF_ERROR(DuplicateColumnsForMultipleRows(
          input, current_output_column, per_row_output_counts_map));
    }

    input.AddColumn(std::move(current_output_column.begin()->second));
  }

  return absl::OkStatus();
}

bool InferenceModel::HasModel() const {
  return !std::holds_alternative<NoModel>(model_);
}

const std::optional<SessionInferenceConfiguration>&
InferenceModel::GetInferenceConfiguration() const {
  return inference_configuration_;
}

}  // namespace confidential_federated_compute::fed_sql
