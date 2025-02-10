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
#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::testing::HasSubstr;

TEST(InferenceModelTest, HasModelNone) {
  InferenceModel inference_model = InferenceModel();
  ASSERT_FALSE(inference_model.HasModel());
}

TEST(InferenceModelTest, HasModelGemma) {
  InferenceModel inference_model = InferenceModel();
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_name: "transcript"
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
      gemma_config { tokenizer_file: "/tmp/tokenizer.json" model: GEMMA_TINY }
    }
    gemma_init_config {
      tokenizer_configuration_id: "tokenizer_configuration_id"
    }
  )pb");
  inference_configuration.gemma_configuration.emplace();
  inference_configuration.gemma_configuration->tokenizer_path =
      "/tmp/tokenizer";
  inference_configuration.gemma_configuration->model_weight_path =
      "/tmp/model_weight";

  ASSERT_TRUE(inference_model.BuildModel(inference_configuration).ok());
  ASSERT_TRUE(inference_model.HasModel());
}

TEST(InferenceModelTest, BuildModelGemmaValidConfig) {
  InferenceModel inference_model = InferenceModel();
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_name: "transcript"
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
      gemma_config {
        tokenizer_file: "/tmp/tokenizer.json"
        model: GEMMA_2B
        model_training: GEMMA_IT
        tensor_type: GEMMA_SFP
      }
    }
    gemma_init_config {
      tokenizer_configuration_id: "tokenizer_configuration_id"
    }
  )pb");
  inference_configuration.gemma_configuration.emplace();
  inference_configuration.gemma_configuration->tokenizer_path =
      "/tmp/tokenizer";
  inference_configuration.gemma_configuration->model_weight_path =
      "/tmp/model_weight";

  ASSERT_TRUE(inference_model.BuildModel(inference_configuration).ok());
}

TEST(InferenceModelTest, BuildModelGemmaInvalidModel) {
  InferenceModel inference_model = InferenceModel();
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_name: "transcript"
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
      gemma_config {
        tokenizer_file: "/tmp/tokenizer.json"
        model: GEMMA_MODEL_UNSPECIFIED
        model_training: GEMMA_IT
        tensor_type: GEMMA_SFP
      }
    }
    gemma_init_config {
      tokenizer_configuration_id: "tokenizer_configuration_id"
    }
  )pb");
  inference_configuration.gemma_configuration.emplace();
  inference_configuration.gemma_configuration->tokenizer_path =
      "/tmp/tokenizer";
  inference_configuration.gemma_configuration->model_weight_path =
      "/tmp/model_weight";

  auto status = inference_model.BuildModel(inference_configuration);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      status.message(),
      HasSubstr("Found invalid InferenceConfiguration.gemma_config.model: 0"));
}

TEST(InferenceModelTest, BuildModelGemmaMissingSessionGemmaConfiguration) {
  InferenceModel inference_model = InferenceModel();
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_name: "transcript"
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
      gemma_config {
        tokenizer_file: "/tmp/tokenizer.json"
        model: GEMMA_2B
        model_training: GEMMA_IT
        tensor_type: GEMMA_SFP
      }
    }
    gemma_init_config {
      tokenizer_configuration_id: "tokenizer_configuration_id"
    }
  )pb");

  auto status = inference_model.BuildModel(inference_configuration);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Missing session Gemma configuration in the model: 2"));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
