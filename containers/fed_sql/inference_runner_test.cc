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
#include "containers/fed_sql/inference_runner.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "containers/common/io/tabular/input.h"
#include "containers/common/local_inference/inference_model.h"
#include "containers/common/local_inference/testing/mocks.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::absl_testing::IsOk;

using ::confidential_federated_compute::local_inference::LlamaCppModel;
using ::confidential_federated_compute::local_inference::
    SessionGemmaCppConfiguration;
using ::confidential_federated_compute::local_inference::
    SessionInferenceConfiguration;
using ::confidential_federated_compute::local_inference::testing::
    MockInferenceModel;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::Not;
using ::testing::Return;
using ::testing::UnorderedElementsAre;

// These tests were ported from inference_model_test.cc when the tabular
// pipeline moved out of InferenceModel. The expected inputs and outputs are
// unchanged; only the mocked seam moved, from RunGemmaCppInference /
// RunLlamaCppInference to the raw GenerateBatch.
class InferenceRunnerTest : public ::testing::Test {
 protected:
  // Helper that makes GenerateBatch return `outputs`, one per prompt.
  void ExpectGenerateBatch(MockInferenceModel& model,
                           std::vector<std::string> outputs) {
    EXPECT_CALL(model, GenerateBatch(_))
        .WillOnce(Invoke([outputs = std::move(outputs)](
                             absl::Span<const std::string> prompts) {
          std::vector<absl::StatusOr<std::string>> results;
          results.reserve(outputs.size());
          for (const std::string& output : outputs) {
            results.push_back(output);
          }
          return results;
        }));
  }
};

TEST_F(InferenceRunnerTest, RunGemmaCppInferenceParserAuto) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" parser: PARSER_AUTO }
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

  // With PARSER_AUTO the runner parses the raw model output as JSON, so the
  // engine returns JSON rather than the final column values.
  auto model = std::make_unique<MockInferenceModel>(inference_configuration);
  ExpectGenerateBatch(*model, {R"({"topic": ["1"]})", R"({"topic": ["2"]})",
                               R"({"topic": ["3"]})"});
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(runner.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 2);
  ASSERT_EQ(output_tensors->at(1).name(), "topic");
  ASSERT_EQ(output_tensors->at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(1).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceRunnerTest, RunGemmaCppInferenceReturnsError) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}" parser: PARSER_AUTO }
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

  auto model = std::make_unique<MockInferenceModel>(inference_configuration);
  EXPECT_CALL(*model, GenerateBatch(_))
      .WillOnce(Invoke([](absl::Span<const std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        results.push_back(absl::InvalidArgumentError("Bad JSON"));
        return results;
      }));
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = runner.RunInference(*input);
  EXPECT_THAT(status, Not(IsOk()));
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("Bad JSON"));
}

TEST_F(InferenceRunnerTest, RunGemmaCppInferenceValidConfig) {
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

  auto model = std::make_unique<MockInferenceModel>(inference_configuration);
  ExpectGenerateBatch(*model, {"1", "2", "3"});
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(runner.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 2);
  ASSERT_EQ(output_tensors->at(1).name(), "topic");
  ASSERT_EQ(output_tensors->at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(1).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceRunnerTest, RunGemmaCppInferenceValidConfigMultipleInputs) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript", "transcript2" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {transcript}, {transcript2}" }
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

  // The runner renders one prompt per row, substituting every input column.
  auto model = std::make_unique<MockInferenceModel>(inference_configuration);
  EXPECT_CALL(*model,
              GenerateBatch(ElementsAre("Hello, one, aol", "Hello, two, bat",
                                        "Hello, three, cat")))
      .WillOnce(Invoke([](absl::Span<const std::string> prompts) {
        return std::vector<absl::StatusOr<std::string>>{"1", "2", "3"};
      }));
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));
  columns.push_back(Tensor({"aol", "bat", "cat"}, "transcript2"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(runner.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 3);
  ASSERT_EQ(output_tensors->at(2).name(), "topic");
  ASSERT_EQ(output_tensors->at(2).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(2).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceRunnerTest, RunInferenceMultipleInferenceTasks) {
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
      inference_task: {
        column_config {
          input_column_names: [ "input" ]
          output_column_name: "output"
        }
        prompt { prompt_template: "Good bye, {{input}}" }
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

  // Tasks run in configuration order, each in its own GenerateBatch call.
  auto model = std::make_unique<MockInferenceModel>(inference_configuration);
  EXPECT_CALL(*model, GenerateBatch(_))
      .WillOnce(Invoke([](absl::Span<const std::string> prompts) {
        return std::vector<absl::StatusOr<std::string>>{"1", "2", "3"};
      }))
      .WillOnce(Invoke([](absl::Span<const std::string> prompts) {
        return std::vector<absl::StatusOr<std::string>>{"1", "2", "3"};
      }));
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(runner.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 4);
  ASSERT_EQ(output_tensors->at(2).name(), "topic");
  ASSERT_EQ(output_tensors->at(2).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(2).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
  ASSERT_EQ(output_tensors->at(3).name(), "output");
  ASSERT_EQ(output_tensors->at(3).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(3).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceRunnerTest, RunInferenceKeepsNonPromptColumns) {
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

  auto model = std::make_unique<MockInferenceModel>(inference_configuration);
  ExpectGenerateBatch(*model, {"1", "2", "3"});
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  // Non-prompt string column.
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input_str_col"));

  // Non-prompt int column.
  columns.push_back(Tensor({1, 2, 3}, "input_int_col"));

  // Prompt column.
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(runner.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 4);
  ASSERT_EQ(output_tensors->at(0).name(), "input_str_col");
  ASSERT_EQ(output_tensors->at(0).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(0).AsSpan<absl::string_view>(),
              UnorderedElementsAre("uno", "dos", "tres"));
  ASSERT_EQ(output_tensors->at(1).name(), "input_int_col");
  ASSERT_EQ(output_tensors->at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(1).AsSpan<int32_t>(),
              UnorderedElementsAre(1, 2, 3));
  ASSERT_EQ(output_tensors->at(2).name(), "transcript");
  ASSERT_EQ(output_tensors->at(2).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(2).AsSpan<absl::string_view>(),
              UnorderedElementsAre("one", "two", "three"));
  ASSERT_EQ(output_tensors->at(3).name(), "topic");
  ASSERT_EQ(output_tensors->at(3).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(3).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceRunnerTest, RunInferenceInputColumnNotFound) {
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

  InferenceRunner runner(
      std::make_unique<MockInferenceModel>(inference_configuration));
  std::vector<Tensor> columns;
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input_str_col"));
  columns.push_back(Tensor({1, 2, 3}, "input_int_col"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = runner.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      status.message(),
      HasSubstr(
          "Couldn't find an input column transcript to run inference on."));
}

TEST_F(InferenceRunnerTest, RunInferenceInputColumnNotFoundMultipleInputs) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript", "transcript2" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript}}, {{transcript2}}" }
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

  InferenceRunner runner(
      std::make_unique<MockInferenceModel>(inference_configuration));
  std::vector<Tensor> columns;
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input_str_col"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = runner.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      status.message(),
      HasSubstr(
          "Couldn't find an input column transcript to run inference on."));
}

TEST_F(InferenceRunnerTest, RunInferenceNoPrompt) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
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

  InferenceRunner runner(
      std::make_unique<MockInferenceModel>(inference_configuration));
  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = runner.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Prompt not found when running inference."));
}

TEST_F(InferenceRunnerTest, RunInferenceNoPromptMultipleInputs) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript", "transcript2" ]
          output_column_name: "topic"
        }
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

  InferenceRunner runner(
      std::make_unique<MockInferenceModel>(inference_configuration));
  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = runner.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Prompt not found when running inference."));
}

TEST_F(InferenceRunnerTest, RunLlamaCppInferenceValidConfigMultipleInputs) {
  // This test is modified based on
  // RunGemmaCppInferenceValidConfigMultipleInputs.
  std::string output_column_name = "topic";

  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript_1", "transcript_2" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Hello, {{transcript_1}}, {{transcript_2}}" }
      }
      gemma_config { model_weight_file: "/tmp/model_weight.gguf" }
    }
    llama_cpp_init_config {
      model_weight_configuration_id: "model_weight_configuration_id"
    }
  )pb");
  inference_configuration.llama_configuration.emplace();
  inference_configuration.llama_configuration->model_weight_path =
      "/tmp/model_weight";

  auto model = std::make_unique<MockInferenceModel>(inference_configuration,
                                                    LlamaCppModel{});
  ExpectGenerateBatch(*model, {"1", "2", "3"});
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript_1"));
  columns.push_back(Tensor({"aol", "bat", "cat"}, "transcript_2"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(runner.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 3);
  ASSERT_EQ(output_tensors->at(2).name(), output_column_name);
  ASSERT_EQ(output_tensors->at(2).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(2).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceRunnerTest, RunLlamaCppInferenceSuccess) {
  // This test is modified based on RunGemmaCppInferenceValidConfig.
  std::string output_column_name = "topic";

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
      gemma_config { model_weight_file: "/tmp/model_weight.gguf" }
    }
    llama_cpp_init_config {
      model_weight_configuration_id: "model_weight_configuration_id"
    }
  )pb");
  inference_configuration.llama_configuration.emplace();
  inference_configuration.llama_configuration->model_weight_path =
      "/tmp/model_weight";

  auto model = std::make_unique<MockInferenceModel>(inference_configuration,
                                                    LlamaCppModel{});
  ExpectGenerateBatch(*model, {"1", "2", "3"});
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(runner.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 2);
  ASSERT_EQ(output_tensors->at(1).name(), output_column_name);
  ASSERT_EQ(output_tensors->at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(1).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceRunnerTest, RunInferenceWithDuplication) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt {
          prompt_template: "Topic for {{transcript}}"
          parser: PARSER_AUTO
        }
      }
    }
    gemma_init_config {}
  )pb");
  inference_configuration.gemma_configuration.emplace();

  // Row 0 yields one value, row 1 yields two, so the input rows are duplicated
  // to match.
  auto model = std::make_unique<MockInferenceModel>(inference_configuration);
  ExpectGenerateBatch(*model,
                      {R"({"topic": ["t1"]})", R"({"topic": ["t2", "t3"]})"});
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"foo", "bar"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(runner.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 2);
  ASSERT_EQ(output_tensors->at(0).name(), "transcript");
  ASSERT_EQ(output_tensors->at(1).name(), "topic");

  EXPECT_THAT(output_tensors->at(0).AsSpan<absl::string_view>(),
              ElementsAre("foo", "bar", "bar"));
  EXPECT_THAT(output_tensors->at(1).AsSpan<absl::string_view>(),
              ElementsAre("t1", "t2", "t3"));
}

TEST_F(InferenceRunnerTest, RunInferenceWithDuplicationAndMultiplePrompts) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt {
          prompt_template: "Topic for {{transcript}}"
          parser: PARSER_AUTO
        }
      }
      inference_task: {
        column_config {
          input_column_names: [ "other_input" ]
          output_column_name: "other_output"
        }
        prompt {
          prompt_template: "Other for {{other_input}}"
          parser: PARSER_AUTO
        }
      }
    }
    gemma_init_config {}
  )pb");
  inference_configuration.gemma_configuration.emplace();

  // Task 1 expands 2 rows to 3, and task 2 then runs over those 3 expanded
  // rows and expands them to 4. This chained behavior is specific to fed_sql:
  // BatchedInferenceFn instead runs every task over the original rows and
  // applies a single Cartesian expansion at the end.
  auto model = std::make_unique<MockInferenceModel>(inference_configuration);
  EXPECT_CALL(*model, GenerateBatch(_))
      .WillOnce(Invoke([](absl::Span<const std::string> prompts) {
        EXPECT_EQ(prompts.size(), 2);
        return std::vector<absl::StatusOr<std::string>>{
            R"({"topic": ["t1"]})", R"({"topic": ["t2", "t3"]})"};
      }))
      .WillOnce(Invoke([](absl::Span<const std::string> prompts) {
        EXPECT_EQ(prompts.size(), 3);
        return std::vector<absl::StatusOr<std::string>>{
            R"({"other_output": ["o1"]})", R"({"other_output": ["o2", "o3"]})",
            R"({"other_output": ["o4"]})"};
      }));
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"foo", "bar"}, "transcript"));
  columns.push_back(Tensor({"a", "b"}, "other_input"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(runner.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 4);
  ASSERT_EQ(output_tensors->at(0).name(), "transcript");
  ASSERT_EQ(output_tensors->at(1).name(), "other_input");
  ASSERT_EQ(output_tensors->at(2).name(), "topic");
  ASSERT_EQ(output_tensors->at(3).name(), "other_output");

  EXPECT_THAT(output_tensors->at(0).AsSpan<absl::string_view>(),
              ElementsAre("foo", "bar", "bar", "bar"));
  EXPECT_THAT(output_tensors->at(1).AsSpan<absl::string_view>(),
              ElementsAre("a", "b", "b", "b"));
  EXPECT_THAT(output_tensors->at(2).AsSpan<absl::string_view>(),
              ElementsAre("t1", "t2", "t2", "t3"));
  EXPECT_THAT(output_tensors->at(3).AsSpan<absl::string_view>(),
              ElementsAre("o1", "o2", "o3", "o4"));
}

TEST_F(InferenceRunnerTest,
       RunInferenceMultipleRowsPreservesMetadataAndPrivacyId) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt {
          prompt_template: "Topic for {{transcript}}"
          parser: PARSER_AUTO
        }
      }
    }
    gemma_init_config {}
  )pb");
  inference_configuration.gemma_configuration.emplace();

  auto model = std::make_unique<MockInferenceModel>(inference_configuration);
  ExpectGenerateBatch(*model,
                      {R"({"topic": ["t1"]})", R"({"topic": ["t2", "t3"]})"});
  InferenceRunner runner(std::move(model));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"foo", "bar"}, "transcript"));

  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), "key_1", "privacy_id_1");
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(runner.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  EXPECT_EQ(input->GetMetadata(), "key_1");
  ASSERT_TRUE(input->GetPrivacyId().has_value());
  EXPECT_EQ(input->GetPrivacyId().value(), "privacy_id_1");
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
