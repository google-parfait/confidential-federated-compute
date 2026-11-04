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

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "containers/common/local_inference/testing/mocks.h"
#include "containers/common/local_inference/testing/test_gguf.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gemma/gemma.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "include/llama.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::local_inference {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

using ::fcp::confidentialcompute::GemmaConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::gcpp::Gemma;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

using ::confidential_federated_compute::local_inference::testing::
    MockInferenceModel;
using ::confidential_federated_compute::local_inference::testing::
    WriteMinimalGemma4Gguf;

// Expected error text when llama_model_load_from_file rejects a weight path.
constexpr absl::string_view kExpectedLlamaLoadError =
    "Unable to load the llama.cpp model.";

// Shared configuration ID for Gemma 4 E4B GGUF test cases.
constexpr absl::string_view kGemma4LlamaWeightConfigId =
    "gemma4_gguf_weight_id";

// Unrecognized GGUF architecture name used to exercise the same unknown
// architecture error path that older llama.cpp revisions hit on "gemma4".
constexpr absl::string_view kUnsupportedGgufArchitecture = "unsupported_arch";

// Captures global llama.cpp log messages for the lifetime of a scope and
// restores the previous logger on destruction.
class ScopedLlamaLogCapture {
 public:
  ScopedLlamaLogCapture() {
    llama_log_get(&prev_callback_, &prev_user_data_);
    llama_log_set(&ScopedLlamaLogCapture::LogCallback, &captured_logs_);
  }

  ~ScopedLlamaLogCapture() { llama_log_set(prev_callback_, prev_user_data_); }

  const std::string& logs() const { return captured_logs_; }

 private:
  static void LogCallback(ggml_log_level /*level*/, const char* text,
                          void* user_data) {
    if (text != nullptr && user_data != nullptr) {
      static_cast<std::string*>(user_data)->append(text);
    }
  }

  ggml_log_callback prev_callback_ = nullptr;
  void* prev_user_data_ = nullptr;
  std::string captured_logs_;
};

SessionInferenceConfiguration CreateGemma4LlamaSessionConfig(
    absl::string_view model_weight_path) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {transcript}" }
      }
      runtime_config { max_generated_tokens: 2 }
      gemma_config {
        model: GEMMA4_E4B
        model_training: GEMMA_IT
        weight_type: WEIGHT_TYPE_GGUF
      }
    }
    llama_cpp_init_config {}
  )pb");
  inference_configuration.initialize_configuration
      .mutable_llama_cpp_init_config()
      ->set_model_weight_configuration_id(
          std::string(kGemma4LlamaWeightConfigId));
  inference_configuration.llama_configuration = SessionLlamaCppConfiguration{
      .model_weight_path = std::string(model_weight_path),
      .n_gpu_layers = 0,
  };
  return inference_configuration;
}

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

TEST_F(InferenceModelTest,
       BuildModelAndGenerateBatchLlamaGemma4MinimalGgufSucceeds) {
  const std::string valid_gguf_path =
      absl::StrCat(::testing::TempDir(), "/minimal_gemma4.gguf");
  ASSERT_TRUE(WriteMinimalGemma4Gguf(valid_gguf_path));

  ScopedLlamaLogCapture log_capture;
  absl::StatusOr<InferenceModel> model = InferenceModel::BuildModel(
      CreateGemma4LlamaSessionConfig(valid_gguf_path));
  ASSERT_THAT(model, IsOk()) << log_capture.logs();
  EXPECT_THAT(log_capture.logs(), Not(HasSubstr("unknown model architecture")));

  std::vector<absl::StatusOr<std::string>> outputs =
      model->GenerateBatch({"hello"});
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_THAT(outputs[0], IsOk());
}

TEST_F(InferenceModelTest,
       BuildModelLlamaUnsupportedArchitectureReturnsInternalError) {
  const std::string unsupported_gguf_path =
      absl::StrCat(::testing::TempDir(), "/unsupported_arch.gguf");
  ASSERT_TRUE(WriteMinimalGemma4Gguf(unsupported_gguf_path,
                                     kUnsupportedGgufArchitecture));

  ScopedLlamaLogCapture log_capture;
  EXPECT_THAT(InferenceModel::BuildModel(
                  CreateGemma4LlamaSessionConfig(unsupported_gguf_path)),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr(kExpectedLlamaLoadError)));
  EXPECT_THAT(log_capture.logs(),
              HasSubstr(absl::StrCat("unknown model architecture: '",
                                     kUnsupportedGgufArchitecture, "'")));
}

TEST_F(InferenceModelTest,
       BuildModelLlamaGemma4E4BInvalidWeightFileReturnsInternalError) {
  const std::string invalid_gguf_path =
      absl::StrCat(::testing::TempDir(), "/invalid_gemma4_e4b.gguf");
  {
    std::ofstream invalid_file(invalid_gguf_path, std::ios::binary);
    ASSERT_TRUE(invalid_file.is_open());
    invalid_file << "not-a-valid-gguf-header";
  }

  SessionInferenceConfiguration inference_configuration =
      CreateGemma4LlamaSessionConfig(invalid_gguf_path);
  EXPECT_THAT(InferenceModel::BuildModel(inference_configuration),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr(kExpectedLlamaLoadError)));

  inference_configuration.llama_configuration->model_weight_path =
      absl::StrCat(::testing::TempDir(), "/missing_gemma4_e4b.gguf");
  EXPECT_THAT(InferenceModel::BuildModel(inference_configuration),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr(kExpectedLlamaLoadError)));
}

TEST_F(InferenceModelTest, PropagatesLlamaCppGemma4E4BRuntimeConfig) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {transcript}" }
      }
      runtime_config { max_generated_tokens: 128 }
      gemma_config {
        model: GEMMA4_E4B
        model_training: GEMMA_IT
        weight_type: WEIGHT_TYPE_GGUF
      }
    }
    llama_cpp_init_config {}
  )pb");
  inference_configuration.initialize_configuration
      .mutable_llama_cpp_init_config()
      ->set_model_weight_configuration_id(
          std::string(kGemma4LlamaWeightConfigId));
  inference_configuration.llama_configuration = SessionLlamaCppConfiguration{
      .model_weight_path =
          absl::StrCat(::testing::TempDir(), "/gemma4_e4b_weights.gguf"),
      .n_gpu_layers = 0,
  };

  MockInferenceModel inference_model(inference_configuration);
  const InferenceInitializeConfiguration& init_config =
      inference_model.GetInferenceConfiguration().initialize_configuration;
  EXPECT_EQ(init_config.inference_config().gemma_config().model(),
            fcp::confidentialcompute::GEMMA4_E4B);
  EXPECT_EQ(init_config.inference_config().gemma_config().weight_type(),
            GemmaConfiguration::WEIGHT_TYPE_GGUF);
  EXPECT_EQ(init_config.llama_cpp_init_config().model_weight_configuration_id(),
            kGemma4LlamaWeightConfigId);
}

}  // namespace
}  // namespace confidential_federated_compute::local_inference
