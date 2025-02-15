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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "compression/io.h"
#include "compression/shared.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
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
using ::fcp::confidentialcompute::GEMMA_2B;
using ::fcp::confidentialcompute::GEMMA_TINY;
using ::fcp::confidentialcompute::GemmaConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::InferenceTask;
using ::fcp::confidentialcompute::Prompt;
using ::gcpp::Gemma;
using ::gcpp::Model;
using ::gcpp::ModelInfo;
using ::gcpp::NestedPools;
using ::gcpp::Path;
using ::gcpp::PromptWrapping;
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
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Found invalid InferenceConfiguration.gemma_config.model: ",
          gemma_config.model()));
  }
  model_info.wrapping = PromptWrapping::GEMMA_IT;
  model_info.weight = Type::kSFP;
  return model_info;
}

}  // namespace

std::unique_ptr<Gemma> InferenceModel::BuildGemmaModel(
    const ModelInfo& model_info,
    const SessionGemmaConfiguration& gemma_config) {
  NestedPools pools(0);
  Path tokenizer_path = Path(gemma_config.tokenizer_path);
  Path weights_path = Path(gemma_config.model_weight_path);
  return std::make_unique<Gemma>(tokenizer_path, weights_path, model_info,
                                 pools);
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
      model_ = BuildGemmaModel(model_info, session_gemma_config);
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
    const std::string& prompt, const absl::string_view& column_value) {
  // TODO: Implement this function.
  return absl::UnimplementedError("Not implemented yet.");
}

absl::Status InferenceModel::RunInference(std::vector<TensorColumn>& columns) {
  if (!HasModel()) {
    return absl::UnimplementedError(
        "Model must be initialized before running inference.");
  }
  std::set<std::string> erase_column_names;
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
                                input_value));
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
        return erase_column_names.find(column.column_schema_.name()) !=
               erase_column_names.end();
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

}  // namespace confidential_federated_compute::fed_sql
