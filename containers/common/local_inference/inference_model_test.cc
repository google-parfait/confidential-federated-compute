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

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "containers/common/local_inference/testing/mocks.h"
#include "gemma/gemma.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::local_inference {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::gcpp::Gemma;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::IsEmpty;
using ::testing::Return;

using ::confidential_federated_compute::local_inference::testing::
    MockInferenceModel;

// NOTE: The generation paths themselves (GenerateBatchGemmaCpp and
// GenerateBatchLlamaCpp) are not unit tested here: gcpp::Gemma and llama_model
// are concrete types loaded from weight files, so there is no seam to fake
// them without real weights. The tests below cover the parts of GenerateBatch
// that are reachable without a loaded model. End-to-end coverage of the
// prompts-in/strings-out contract lives in
// //containers/local_inference:local_batched_inference_engine_test.
class InferenceModelTest : public ::testing::Test {};

TEST_F(InferenceModelTest, GenerateBatchWithNoPromptsReturnsNoResults) {
  MockInferenceModel inference_model;
  EXPECT_THAT(inference_model.InferenceModel::GenerateBatch({}), IsEmpty());
}

TEST_F(InferenceModelTest, BuildModelGemmaMissingSessionGemmaCppConfiguration) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
    }
    gemma_init_config {
      tokenizer_configuration_id: "tokenizer_configuration_id"
      model_weight_configuration_id: "model_weight_configuration_id"
    }
  )pb");

  auto status = InferenceModel::BuildModel(inference_configuration).status();
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  // gemma_configuration is missing from inference_configuration.
  ASSERT_THAT(status.message(),
              HasSubstr("Missing session gemma.cpp configuration"));
}

TEST_F(InferenceModelTest, BuildModelGemmaInvalidSharedModel) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
    }
    gemma_init_config {
      tokenizer_configuration_id: "tokenizer_configuration_id"
      model_weight_configuration_id: "model_weight_configuration_id"
    }
  )pb");
  inference_configuration.shared_gemma_model =
      std::make_shared<GemmaCppModel>();

  auto status = InferenceModel::BuildModel(inference_configuration).status();
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("SessionInferenceConfiguration must contain a valid "
                        "shared Gemma model"));
}

TEST_F(InferenceModelTest, BuildModelLlamaMissingSessionLlamaCppConfiguration) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
    }
    llama_cpp_init_config {
      model_weight_configuration_id: "model_weight_configuration_id"
    }
  )pb");

  auto status = InferenceModel::BuildModel(inference_configuration).status();
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  // llama_configuration is missing from inference_configuration.
  ASSERT_THAT(status.message(),
              HasSubstr("Missing session llama.cpp configuration"));
}

TEST_F(InferenceModelTest, PropagatesRuntimeConfig) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" }
      }
      runtime_config {
        seq_len: 10000
        max_prompt_size: 100
        max_generated_tokens: 50
        temperature_diff: -0.5
      }
      gemma_config { tokenizer_file: "/tmp/tokenizer.json" }
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

  MockInferenceModel inference_model(inference_configuration);
  const auto& config = inference_model.GetInferenceConfiguration()
                           .initialize_configuration.inference_config();
  EXPECT_EQ(config.runtime_config().seq_len(), 10000);
  EXPECT_EQ(config.runtime_config().max_prompt_size(), 100);
  EXPECT_EQ(config.runtime_config().max_generated_tokens(), 50);
  EXPECT_EQ(config.runtime_config().temperature_diff(), -0.5);
}

}  // namespace
}  // namespace confidential_federated_compute::local_inference
