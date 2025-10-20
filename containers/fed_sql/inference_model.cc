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

#include <regex>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "fcp/base/status_converters.h"
#include "gemma/gemma_args.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::fcp::confidentialcompute::ColumnConfiguration;
using ::fcp::confidentialcompute::ColumnSchema;
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
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

constexpr size_t kMaxPromptSize = 10000;

// Apply a regex matching to the given text. Returns only the first match. If
// no match is found, returns the original text.
std::string RegexMatch(const std::string& text, const std::regex& regex) {
  std::smatch match;
  if (std::regex_match(text, match, regex) && match.size() > 1) {
    return match[1];
  }
  return text;
}

}  // namespace

void InferenceModel::BuildGemmaCppModel(
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
}

absl::Status InferenceModel::BuildModel(
    const SessionInferenceConfiguration& inference_configuration) {
  inference_configuration_ = inference_configuration;
  switch (inference_configuration.initialize_configuration
              .model_init_config_case()) {
    case InferenceInitializeConfiguration::kGemmaInitConfig: {
      if (!inference_configuration.gemma_configuration.has_value()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Missing session Gemma configuration"));
      }
      SessionGemmaCppConfiguration session_gemma_config =
          inference_configuration.gemma_configuration.value();
      model_.emplace<GemmaCppModel>();
      BuildGemmaCppModel(session_gemma_config);
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

absl::StatusOr<std::string> InferenceModel::RunGemmaCppInference(
    const std::string& prompt, const absl::string_view& column_value,
    const std::string& column_name) {
  std::string combined_prompt(prompt);
  std::string old_value = absl::StrCat("{", column_name, "}");
  size_t pos;
  while ((pos = combined_prompt.find(old_value)) != std::string::npos) {
    combined_prompt.replace(pos, old_value.size(), column_value);
  }

  size_t generated = 0;
  GemmaCppModel& gemma_model = std::get<GemmaCppModel>(model_);
  Gemma* gemma = gemma_model.gemma_.get();
  KVCache kv_cache(gemma->Config(), gemma->Inference(),
                   gemma_model.ctx_->allocator);
  RuntimeConfig inference_runtime_config =
      inference_configuration_->initialize_configuration.inference_config()
          .runtime_config();
  size_t max_prompt_size =
      inference_runtime_config.max_prompt_size() > 0 ?: kMaxPromptSize;
  if (combined_prompt.size() > max_prompt_size) {
    combined_prompt.resize(max_prompt_size);
  }
  const std::vector<int> tokens = gcpp::WrapAndTokenize(
      gemma->Tokenizer(), gemma->ChatTemplate(), gemma->Config().wrapping,
      generated, combined_prompt);
  const size_t prompt_size = tokens.size();
  std::stringstream output_stream;
  auto stream_token = [&gemma, &output_stream, &generated, &prompt_size](
                          int token, float) {
    ++generated;
    if (generated >= prompt_size && !gemma->Config().IsEOS(token)) {
      std::string token_text;
      FCP_CHECK(gemma->Tokenizer().Decode({token}, &token_text));
      output_stream << token_text;
    }
    return true;
  };
  TimingInfo timing_info;
  std::mt19937 gen;
  std::random_device rd;
  gen.seed(rd());
  size_t max_generated_tokens =
      inference_runtime_config.max_generated_tokens() > 0 ?: 1024;
  ::gcpp::RuntimeConfig runtime_config = {
      .max_generated_tokens = max_generated_tokens,
      .temperature = 1.0,
      .gen = &gen,
      .verbosity = 0,
      .stream_token = stream_token,
  };
  gemma->Generate(runtime_config, tokens, 0, kv_cache, *gemma_model.env_,
                  timing_info);
  return output_stream.str();
}

absl::Status InferenceModel::RunInference(std::vector<Tensor>& columns) {
  if (!HasModel()) {
    return absl::UnimplementedError(
        "Model must be initialized before running inference.");
  }
  absl::flat_hash_set<std::string> erase_column_names;
  for (const auto& inference_task :
       inference_configuration_->initialize_configuration.inference_config()
           .inference_task()) {
    if (!inference_task.has_prompt()) {
      // Only prompt-based inference is supported.
      return absl::InvalidArgumentError(
          "Prompt not found when running inference.");
    }
    const std::string& input_column_name =
        inference_task.column_config().input_column_name();
    const std::string& output_column_name =
        inference_task.column_config().output_column_name();

    // Iterate through the columns to find the input column.
    Tensor output_column;
    const auto it = find_if(columns.begin(), columns.end(),
                            [&input_column_name](const Tensor& column) {
                              return column.name() == input_column_name;
                            });
    if (it == columns.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Couldn't find an input column ", input_column_name,
                       " to run inference on."));
    }
    const Tensor& input_column = *it;
    if (input_column.dtype() != DataType::DT_STRING) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input column ", input_column_name, " is not of type STRING."));
    }
    if (input_column.shape().dim_sizes().size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input column ", input_column_name,
                       " is not a one-dimensional tensor."));
    }
    int64_t tensor_size = input_column.shape().dim_sizes()[0];
    std::unique_ptr<MutableStringData> output_string_data =
        std::make_unique<MutableStringData>(tensor_size);
    std::unique_ptr<std::regex> regex;
    if (!inference_task.prompt().regex().empty()) {
      regex = std::make_unique<std::regex>(inference_task.prompt().regex());
    }
    for (const auto& input_value : input_column.AsSpan<absl::string_view>()) {
      std::string output_string;
      if (std::holds_alternative<GemmaCppModel>(model_)) {
        FCP_ASSIGN_OR_RETURN(
            output_string,
            RunGemmaCppInference(inference_task.prompt().prompt_template(),
                                 input_value, input_column_name));
      } else {
        return absl::UnimplementedError(
            absl::StrCat("Unsupported inference model type."));
      }
      if (regex) {
        output_string = RegexMatch(output_string, *regex);
      }
      output_string_data->Add(std::move(output_string));
      // We can't remove the input column here yet as multiple prompts may rely
      // on the same input columns.
      erase_column_names.insert(input_column_name);
    }

    FCP_ASSIGN_OR_RETURN(
        Tensor out_tensor,
        Tensor::Create(DataType::DT_STRING, TensorShape({tensor_size}),
                       std::move(output_string_data), output_column_name));

    columns.push_back(std::move(out_tensor));
  }

  auto new_end =
      std::remove_if(columns.begin(), columns.end(),
                     [&erase_column_names](const Tensor& column) {
                       return erase_column_names.contains(column.name());
                     });
  columns.erase(new_end, columns.end());
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
