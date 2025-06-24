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

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "compression/io.h"
#include "compression/shared.h"
#include "fcp/base/status_converters.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "util/threading.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::confidential_federated_compute::sql::TensorColumn;
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
using ::gcpp::Allocator;
using ::gcpp::BoundedTopology;
using ::gcpp::EOS_ID;
using ::gcpp::Gemma;
using ::gcpp::KVCache;
using ::gcpp::MatMulEnv;
using ::gcpp::Model;
using ::gcpp::ModelInfo;
using ::gcpp::NestedPools;
using ::gcpp::Path;
using ::gcpp::PromptWrapping;
using ::gcpp::RuntimeConfig;
using ::gcpp::TimingInfo;
using ::gcpp::Type;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_STRING;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

absl::StatusOr<ModelInfo> GetGemmaModelInfo(
    const GemmaConfiguration& gemma_config) {
  ModelInfo model_info;
  switch (gemma_config.model()) {
    case GEMMA_TINY: {
      model_info.model = Model::GEMMA_TINY;
      break;
    }
    case GEMMA_2B: {
      model_info.model = Model::GEMMA_2B;
      break;
    }
    case GEMMA2_2B: {
      model_info.model = Model::GEMMA2_2B;
      break;
    }
    case GEMMA_7B: {
      model_info.model = Model::GEMMA_7B;
      break;
    }
    case GEMMA2_9B: {
      model_info.model = Model::GEMMA2_9B;
      break;
    }
    case GEMMA3_1B: {
      model_info.model = Model::GEMMA3_1B;
      break;
    }
    case GEMMA3_4B: {
      model_info.model = Model::GEMMA3_4B;
      break;
    }
    case GEMMA3_12B: {
      model_info.model = Model::GEMMA3_12B;
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Found invalid InferenceConfiguration.gemma_config.model: ",
          gemma_config.model()));
  }
  switch (gemma_config.model_training()) {
    case GEMMA_IT: {
      model_info.wrapping = PromptWrapping::GEMMA_IT;
      break;
    }
    case GEMMA_PT: {
      model_info.wrapping = PromptWrapping::GEMMA_PT;
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Found invalid "
                       "InferenceConfiguration.gemma_config.model_training: ",
                       gemma_config.model_training()));
  }
  switch (gemma_config.tensor_type()) {
    case GEMMA_F32: {
      model_info.weight = Type::kF32;
      break;
    }
    case GEMMA_SFP: {
      model_info.weight = Type::kSFP;
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Found invalid "
                       "InferenceConfiguration.gemma_config.tensor_type: ",
                       gemma_config.tensor_type()));
  }
  return model_info;
}

}  // namespace

void InferenceModel::BuildGemmaModel(
    const ModelInfo& model_info,
    const SessionGemmaConfiguration& gemma_config) {
  BoundedTopology topology;
  Allocator::Init(topology);
  GemmaModel& gemma_model = std::get<GemmaModel>(model_);
  gemma_model.pools_ = std::make_unique<NestedPools>(topology);
  gemma_model.env_ = std::make_unique<MatMulEnv>(topology, *gemma_model.pools_);
  Path tokenizer_path = Path(gemma_config.tokenizer_path);
  Path weights_path = Path(gemma_config.model_weight_path);
  gemma_model.gemma_ = std::make_unique<Gemma>(tokenizer_path, weights_path,
                                               model_info, *gemma_model.env_);
}

absl::Status InferenceModel::BuildModel(
    const SessionInferenceConfiguration& inference_configuration) {
  inference_configuration_ = inference_configuration;
  switch (inference_configuration.initialize_configuration
              .model_init_config_case()) {
    case InferenceInitializeConfiguration::kGemmaInitConfig: {
      const GemmaConfiguration& gemma_config =
          inference_configuration.initialize_configuration.inference_config()
              .gemma_config();
      if (!inference_configuration.gemma_configuration.has_value()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Missing session Gemma configuration in the model: ",
                         gemma_config.model()));
      }
      FCP_ASSIGN_OR_RETURN(ModelInfo model_info,
                           GetGemmaModelInfo(gemma_config));
      SessionGemmaConfiguration session_gemma_config =
          inference_configuration.gemma_configuration.value();
      model_.emplace<GemmaModel>();
      BuildGemmaModel(model_info, session_gemma_config);
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

absl::StatusOr<std::string> InferenceModel::RunGemmaInference(
    const std::string& prompt, const absl::string_view& column_value,
    const std::string& column_name) {
  std::string combined_prompt(prompt);
  std::string old_value = absl::StrCat("{", column_name, "}");
  size_t pos;
  while ((pos = combined_prompt.find(old_value)) != std::string::npos) {
    combined_prompt.replace(pos, old_value.size(), column_value);
  }

  const size_t prefill_tbatch_size = 64;
  size_t generated = 0;
  GemmaModel& gemma_model = std::get<GemmaModel>(model_);
  Gemma* gemma = gemma_model.gemma_.get();
  KVCache kv_cache =
      KVCache::Create(gemma->GetModelConfig(), prefill_tbatch_size);
  const std::vector<int> tokens = gcpp::WrapAndTokenize(
      gemma->Tokenizer(), gemma->Info(), generated, combined_prompt);
  const size_t prompt_size = tokens.size();
  std::stringstream output_stream;
  auto stream_token = [&gemma, &output_stream, &generated, &prompt_size](
                          int token, float) {
    ++generated;
    if (generated >= prompt_size && !gemma->GetModelConfig().IsEOS(token)) {
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
  RuntimeConfig runtime_config = {
      .max_generated_tokens = 1024,
      .temperature = 1.0,
      .gen = &gen,
      .verbosity = 0,
      .stream_token = stream_token,
  };
  {
    // RunGemmaInference can be called within different sessions concurrently,
    // but the gemma->Generate method is not thread-safe.
    absl::MutexLock l(&mutex_);
    gemma->Generate(runtime_config, tokens, 0, kv_cache, timing_info);
  }
  return output_stream.str();
}

absl::Status InferenceModel::RunInference(std::vector<TensorColumn>& columns) {
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
    TensorColumn output_column;
    const auto it =
        find_if(columns.begin(), columns.end(),
                [&input_column_name](const TensorColumn& column) {
                  return column.column_schema_.name() == input_column_name;
                });
    if (it == columns.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Couldn't find an input column ", input_column_name,
                       " to run inference on."));
    }
    const TensorColumn& input_column = *it;
    if (input_column.column_schema_.type() !=
        ExampleQuerySpec_OutputVectorSpec_DataType_STRING) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input column ", input_column_name, " is not of type STRING."));
    }
    if (input_column.tensor_.shape().dim_sizes().size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input column ", input_column_name,
                       " is not a one-dimensional tensor."));
    }
    int64_t tensor_size = input_column.tensor_.shape().dim_sizes()[0];
    std::unique_ptr<MutableStringData> output_string_data =
        std::make_unique<MutableStringData>(tensor_size);
    for (const auto& input_value :
         input_column.tensor_.AsSpan<absl::string_view>()) {
      std::string output_string;
      ModelType model_type = GetModelType();
      switch (model_type) {
        case ModelType::kGemma: {
          FCP_ASSIGN_OR_RETURN(
              output_string,
              RunGemmaInference(inference_task.prompt().prompt_template(),
                                input_value, input_column_name));
          break;
        }
        default:
          return absl::UnimplementedError(
              absl::StrCat("Unsupported model type: ", model_type));
      }
      output_string_data->Add(std::move(output_string));
      // We can't remove the input column here yet as multiple prompts may rely
      // on the same input columns.
      erase_column_names.insert(input_column_name);
    }

    FCP_ASSIGN_OR_RETURN(
        Tensor out_tensor,
        Tensor::Create(DataType::DT_STRING, TensorShape({tensor_size}),
                       std::move(output_string_data)));

    ColumnSchema out_col_schema;
    out_col_schema.set_name(output_column_name);
    out_col_schema.set_type(ExampleQuerySpec_OutputVectorSpec_DataType_STRING);
    FCP_ASSIGN_OR_RETURN(
        output_column,
        TensorColumn::Create(out_col_schema, std::move(out_tensor)));

    columns.push_back(std::move(output_column));
  }

  auto new_end = std::remove_if(
      columns.begin(), columns.end(),
      [&erase_column_names](const TensorColumn& column) {
        return erase_column_names.contains(column.column_schema_.name());
      });
  columns.erase(new_end, columns.end());
  return absl::OkStatus();
}

ModelType InferenceModel::GetModelType() const {
  return static_cast<ModelType>(model_.index());
}

bool InferenceModel::HasModel() const {
  return GetModelType() != ModelType::kNone;
}

const std::optional<SessionInferenceConfiguration>&
InferenceModel::GetInferenceConfiguration() const {
  return inference_configuration_;
}

}  // namespace confidential_federated_compute::fed_sql
