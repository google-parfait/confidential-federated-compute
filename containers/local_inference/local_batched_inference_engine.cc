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

#include "local_batched_inference_engine.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "containers/common/local_inference/inference_model.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"

namespace confidential_federated_compute::local_inference {

namespace {

using ::confidential_federated_compute::fns::WriteConfigurationMap;
using ::fcp::confidentialcompute::GemmaInitializeConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::LlamaCppInitializeConfiguration;
using ::fcp::confidentialcompute::RuntimeConfig;

// Resolves a configuration id to the path of the blob that was streamed for it
// during StreamInitialize.
absl::StatusOr<std::string> LookUpConfigurationPath(
    const WriteConfigurationMap& write_configuration_map,
    const std::string& configuration_id, absl::string_view description) {
  auto it = write_configuration_map.find(configuration_id);
  if (it == write_configuration_map.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected ", description, " configuration id ", configuration_id,
        " is missing in WriteConfigurationRequest."));
  }
  return it->second;
}

// Turns the streamed configuration blobs plus the init config into the
// SessionInferenceConfiguration that InferenceModel expects.
absl::StatusOr<SessionInferenceConfiguration>
BuildSessionInferenceConfiguration(
    const InferenceInitializeConfiguration& init_config,
    const WriteConfigurationMap& write_configuration_map) {
  SessionInferenceConfiguration session_config;
  session_config.initialize_configuration = init_config;

  switch (init_config.model_init_config_case()) {
    case InferenceInitializeConfiguration::kGemmaInitConfig: {
      const GemmaInitializeConfiguration& gemma_init_config =
          init_config.gemma_init_config();
      SessionGemmaCppConfiguration gemma_config;
      ABSL_ASSIGN_OR_RETURN(gemma_config.tokenizer_path,
                            LookUpConfigurationPath(
                                write_configuration_map,
                                gemma_init_config.tokenizer_configuration_id(),
                                "gemma.cpp tokenizer"));
      ABSL_ASSIGN_OR_RETURN(
          gemma_config.model_weight_path,
          LookUpConfigurationPath(
              write_configuration_map,
              gemma_init_config.model_weight_configuration_id(),
              "gemma.cpp weight"));
      session_config.gemma_configuration = std::move(gemma_config);
      break;
    }
    case InferenceInitializeConfiguration::kLlamaCppInitConfig: {
      const LlamaCppInitializeConfiguration& llama_init_config =
          init_config.llama_cpp_init_config();
      SessionLlamaCppConfiguration llama_config;
      ABSL_ASSIGN_OR_RETURN(
          llama_config.model_weight_path,
          LookUpConfigurationPath(
              write_configuration_map,
              llama_init_config.model_weight_configuration_id(),
              "llama.cpp weight"));
      session_config.llama_configuration = std::move(llama_config);
      break;
    }
    default:
      return absl::FailedPreconditionError(absl::StrCat(
          "InferenceInitializeConfiguration.model_init_config must be set. "
          "Got model_init_config_case: ",
          init_config.model_init_config_case()));
  }
  return session_config;
}

}  // namespace

LocalBatchedInferenceEngine::LocalBatchedInferenceEngine(
    std::shared_ptr<InferenceModel> inference_model)
    : inference_model_(std::move(inference_model)) {}

std::vector<absl::StatusOr<std::string>>
LocalBatchedInferenceEngine::DoBatchedInference(
    std::vector<std::string> prompts) {
  if (prompts.empty()) {
    return {};
  }
  if (inference_model_ == nullptr) {
    std::vector<absl::StatusOr<std::string>> results;
    results.reserve(prompts.size());
    for (size_t i = 0; i < prompts.size(); ++i) {
      results.push_back(
          absl::FailedPreconditionError("Inference model is not initialized."));
    }
    return results;
  }
  // `prompts` is already fully rendered and already bounded by
  // RuntimeConfig.max_batch_size by BatchedInferenceFn, so no templating and
  // no further chunking belong here.
  //
  // Note that this API structure may make it harder to apply optimizations
  // like prompt prefix caching in the future. We should revisit that at some
  // point.
  return inference_model_->GenerateBatch(prompts);
}

absl::StatusOr<std::shared_ptr<
    ::confidential_federated_compute::inference::BatchedInferenceEngine>>
LocalBatchedInferenceEngineProvider::GetEngineForInferenceConfig(
    const InferenceInitializeConfiguration& init_config,
    const WriteConfigurationMap& write_configuration_map) {
  const RuntimeConfig& runtime_config =
      init_config.inference_config().runtime_config();
  if (runtime_config.model_mode() ==
      RuntimeConfig::MODEL_MODE_MODEL_PER_SESSION) {
    return absl::UnimplementedError(
        "MODEL_MODE_MODEL_PER_SESSION is not supported by the local_inference "
        "container. Model weights are loaded once and shared read-only across "
        "BatchedInferenceFn instances.");
  }

  ABSL_ASSIGN_OR_RETURN(
      SessionInferenceConfiguration session_config,
      BuildSessionInferenceConfiguration(init_config, write_configuration_map));

  ABSL_ASSIGN_OR_RETURN(InferenceModel model,
                        InferenceModel::BuildModel(session_config));

  return std::make_shared<LocalBatchedInferenceEngine>(
      std::make_shared<InferenceModel>(std::move(model)));
}

}  // namespace confidential_federated_compute::local_inference
