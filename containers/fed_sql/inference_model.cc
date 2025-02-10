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

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "compression/io.h"
#include "compression/shared.h"
#include "fcp/base/status_converters.h"
#include "gemma/common.h"
#include "gemma/configs.h"
#include "util/threading.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::confidential_federated_compute::sql::TensorColumn;
using ::fcp::confidentialcompute::GEMMA2_2B;
using ::fcp::confidentialcompute::GEMMA_2B;
using ::fcp::confidentialcompute::GEMMA_TINY;
using ::fcp::confidentialcompute::GemmaConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;

using ::gcpp::Gemma;
using ::gcpp::Model;
using ::gcpp::ModelInfo;
using ::gcpp::NestedPools;
using ::gcpp::Path;
using ::gcpp::PromptWrapping;
using ::gcpp::Type;

absl::StatusOr<std::unique_ptr<Gemma>> BuildGemmaModel(
    const SessionInferenceConfiguration& inference_configuration) {
  ModelInfo model_info;
  const GemmaConfiguration& gemma_config =
      inference_configuration.initialize_configuration.inference_config()
          .gemma_config();
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
  NestedPools pools(0);
  if (!inference_configuration.gemma_configuration.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing session Gemma configuration in the model: ",
                     gemma_config.model()));
  }
  const SessionGemmaConfiguration& session_gemma_config =
      inference_configuration.gemma_configuration.value();
  Path tokenizer_path = Path(session_gemma_config.tokenizer_path);
  Path weights_path = Path(session_gemma_config.model_weight_path);
  // TODO: Return
  // std::make_unique<Gemma>(tokenizer_path, weights_path, model_info, pools)
  // once tokenizer and weights are populated in GemmaConfiguration.
  return nullptr;
}

}  // namespace

absl::Status InferenceModel::BuildModel(
    const SessionInferenceConfiguration& inference_configuration) {
  inference_configuration_ = inference_configuration;
  switch (inference_configuration.initialize_configuration
              .model_init_config_case()) {
    case InferenceInitializeConfiguration::kGemmaInitConfig: {
      model_type_ = ModelType::kGemma;
      FCP_ASSIGN_OR_RETURN(gemma_model_,
                           BuildGemmaModel(inference_configuration));
      break;
    }
    default:
      model_type_ = ModelType::kNone;
      return absl::UnimplementedError(
          absl::StrCat("Unsupported model_init_config_case: ",
                       inference_configuration.initialize_configuration
                           .model_init_config_case()));
      break;
  }
  return absl::OkStatus();
}

void InferenceModel::RunInference(std::vector<TensorColumn>& columns) {
  // TODO: Call an actual model inference.
}

bool InferenceModel::HasModel() const {
  return model_type_ != ModelType::kNone;
}

}  // namespace confidential_federated_compute::fed_sql
