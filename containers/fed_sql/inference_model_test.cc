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
#include "absl/status/status_matchers.h"
#include "containers/common/io/tabular/input.h"
#include "containers/fed_sql/testing/mocks.h"
#include "gemma/gemma.h"
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
using ::absl_testing::IsOkAndHolds;

using ::confidential_federated_compute::fed_sql::testing::MockInferenceModel;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::Prompt;
using ::gcpp::Gemma;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::_;
using ::testing::ByMove;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::Not;
using ::testing::Property;
using ::testing::Return;
using ::testing::UnorderedElementsAre;

class InferenceModelTest : public ::testing::Test {
 protected:
  // Helper to set expectations on RunGemmaCppInference.
  void ExpectGemmaCppInference(
      MockInferenceModel& inference_model,
      const std::vector<std::string>& input_names,
      const std::string& output_name,
      absl::StatusOr<MockInferenceModel::InferenceOutput> output) {
    EXPECT_CALL(inference_model, RunGemmaCppInference(_, _, _, Eq(output_name)))
        .WillOnce(Invoke([output = std::move(output)]() mutable {
          return std::move(output);
        }));
  }
};

TEST_F(InferenceModelTest, RunGemmaCppInferenceParserAuto) {
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

  MockInferenceModel inference_model(inference_configuration);
  MockInferenceModel::InferenceOutput output;
  output.tensor = Tensor({"1", "2", "3"}, "topic");
  output.per_row_output_counts = {1, 1, 1};
  EXPECT_CALL(
      inference_model,
      RunGemmaCppInference(Property(&Prompt::parser, Eq(Prompt::PARSER_AUTO)),
                           _, _, Eq("topic")))
      .WillOnce(Invoke([output = std::move(output)]() mutable {
        return std::move(output);
      }));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 2);
  ASSERT_EQ(output_tensors->at(1).name(), "topic");
  ASSERT_EQ(output_tensors->at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(1).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceModelTest, RunGemmaCppInferenceReturnsError) {
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

  MockInferenceModel inference_model(inference_configuration);
  EXPECT_CALL(inference_model, RunGemmaCppInference(_, _, _, _))
      .WillOnce(
          Invoke([]() { return absl::InvalidArgumentError("Bad JSON"); }));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model.RunInference(*input);
  EXPECT_THAT(status, Not(IsOk()));
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("Bad JSON"));
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

TEST_F(InferenceModelTest, RunGemmaCppInferenceValidConfig) {
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

  MockInferenceModel inference_model(inference_configuration);
  MockInferenceModel::InferenceOutput output;
  output.tensor = Tensor({"1", "2", "3"}, "topic");
  output.per_row_output_counts = {1, 1, 1};
  ExpectGemmaCppInference(inference_model, {"transcript"}, "topic",
                          std::move(output));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 2);
  ASSERT_EQ(output_tensors->at(1).name(), "topic");
  ASSERT_EQ(output_tensors->at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(1).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceModelTest, RunGemmaCppInferenceValidConfigMultipleInputs) {
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

  MockInferenceModel inference_model(inference_configuration);
  MockInferenceModel::InferenceOutput output;
  output.tensor = Tensor({"1", "2", "3"}, "topic");
  output.per_row_output_counts = {1, 1, 1};
  ExpectGemmaCppInference(inference_model, {"transcript", "transcript2"},
                          "topic", std::move(output));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));
  columns.push_back(Tensor({"aol", "bat", "cat"}, "transcript2"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 3);
  ASSERT_EQ(output_tensors->at(2).name(), "topic");
  ASSERT_EQ(output_tensors->at(2).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(2).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceModelTest, RunInferenceMultipleInferenceTasks) {
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

  MockInferenceModel inference_model(inference_configuration);
  MockInferenceModel::InferenceOutput topic_output;
  topic_output.tensor = Tensor({"1", "2", "3"}, "topic");
  topic_output.per_row_output_counts = {1, 1, 1};
  ExpectGemmaCppInference(inference_model, {"transcript"}, "topic",
                          std::move(topic_output));

  MockInferenceModel::InferenceOutput output_output;
  output_output.tensor = Tensor({"1", "2", "3"}, "output");
  output_output.per_row_output_counts = {1, 1, 1};
  ExpectGemmaCppInference(inference_model, {"input"}, "output",
                          std::move(output_output));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
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

TEST_F(InferenceModelTest, RunInferenceKeepsNonPromptColumns) {
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

  MockInferenceModel inference_model(inference_configuration);
  MockInferenceModel::InferenceOutput output;
  output.tensor = Tensor({"1", "2", "3"}, "topic");
  output.per_row_output_counts = {1, 1, 1};
  ExpectGemmaCppInference(inference_model, {"transcript"}, "topic",
                          std::move(output));

  std::vector<Tensor> columns;
  // Non-prompt string column.
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input_str_col"));

  // Non-prompt int column.
  columns.push_back(Tensor({1, 2, 3}, "input_int_col"));

  // Prompt column.
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
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

TEST_F(InferenceModelTest, RunInferenceInputColumnNotFound) {
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

  MockInferenceModel inference_model(inference_configuration);
  std::vector<Tensor> columns;
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input_str_col"));
  columns.push_back(Tensor({1, 2, 3}, "input_int_col"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      status.message(),
      HasSubstr(
          "Couldn't find an input column transcript to run inference on."));
}

TEST_F(InferenceModelTest, RunInferenceInputColumnNotFoundMultipleInputs) {
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

  MockInferenceModel inference_model(inference_configuration);
  std::vector<Tensor> columns;
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input_str_col"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      status.message(),
      HasSubstr(
          "Couldn't find an input column transcript to run inference on."));
}

TEST_F(InferenceModelTest, RunInferenceNoPrompt) {
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

  MockInferenceModel inference_model(inference_configuration);
  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Prompt not found when running inference."));
}

TEST_F(InferenceModelTest, RunInferenceNoPromptMultipleInputs) {
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

  MockInferenceModel inference_model(inference_configuration);
  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Prompt not found when running inference."));
}

TEST_F(InferenceModelTest, RunInferenceWithRuntimeConfigFlags) {
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
  EXPECT_CALL(inference_model, RunGemmaCppInference(_, _, _, _))
      .WillOnce([&inference_model](const Prompt&, const Input&,
                                   absl::Span<const size_t>,
                                   const std::string&) mutable
                    -> absl::StatusOr<MockInferenceModel::InferenceOutput> {
        const auto& config = inference_model.GetInferenceConfiguration()
                                 .initialize_configuration.inference_config();
        EXPECT_EQ(config.runtime_config().seq_len(), 10000);
        EXPECT_EQ(config.runtime_config().max_prompt_size(), 100);
        EXPECT_EQ(config.runtime_config().max_generated_tokens(), 50);
        EXPECT_EQ(config.runtime_config().temperature_diff(), -0.5);
        MockInferenceModel::InferenceOutput output;
        output.tensor = Tensor({"r1", "r2", "r3"}, "topic");
        output.per_row_output_counts = {1, 1, 1};
        return std::move(output);
      });

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 2);
  ASSERT_EQ(output_tensors->at(1).name(), "topic");
  ASSERT_EQ(output_tensors->at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(1).AsSpan<absl::string_view>(),
              UnorderedElementsAre("r1", "r2", "r3"));
}

TEST_F(InferenceModelTest, RunLlamaCppInferenceValidConfigMultipleInputs) {
  // This test is modified based on
  // RunGemmaCppInferenceValidConfigMultipleInputs.
  std::string output_column_name = "topic";
  // Mock the RunLlamaCppInferencePerRow function to return the output tensor.
  MockInferenceModel::InferenceOutput output;
  output.tensor = Tensor({"1", "2", "3"}, output_column_name);
  output.per_row_output_counts = {1, 1, 1};

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

  MockInferenceModel inference_model(inference_configuration, LlamaCppModel{});
  EXPECT_CALL(inference_model,
              RunLlamaCppInference(_, _, _, Eq(output_column_name)))
      .WillOnce(Invoke([output = std::move(output)]() mutable {
        return std::move(output);
      }));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript_1"));
  columns.push_back(Tensor({"aol", "bat", "cat"}, "transcript_2"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 3);
  ASSERT_EQ(output_tensors->at(2).name(), output_column_name);
  ASSERT_EQ(output_tensors->at(2).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(2).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceModelTest, RunLlamaCppInferenceSuccess) {
  // This test is modified based on RunGemmaCppInferenceValidConfig.
  std::string output_column_name = "topic";
  // Mock the RunLlamaCppInferencePerRow function to return the output tensor.
  MockInferenceModel::InferenceOutput output;
  output.tensor = Tensor({"1", "2", "3"}, output_column_name);
  output.per_row_output_counts = {1, 1, 1};

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

  MockInferenceModel inference_model(inference_configuration, LlamaCppModel{});
  EXPECT_CALL(inference_model,
              RunLlamaCppInference(_, _, _, Eq(output_column_name)))
      .WillOnce(Invoke([output = std::move(output)]() mutable {
        return std::move(output);
      }));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  ASSERT_EQ(output_tensors->size(), 2);
  ASSERT_EQ(output_tensors->at(1).name(), output_column_name);
  ASSERT_EQ(output_tensors->at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(output_tensors->at(1).AsSpan<absl::string_view>(),
              UnorderedElementsAre("1", "2", "3"));
}

TEST_F(InferenceModelTest, RunInferenceWithDuplication) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Topic for {{transcript}}" }
      }
    }
    gemma_init_config {}
  )pb");
  inference_configuration.gemma_configuration.emplace();

  MockInferenceModel inference_model(inference_configuration);
  MockInferenceModel::InferenceOutput topic_output;
  topic_output.tensor = Tensor({"t1", "t2", "t3"}, "topic");
  topic_output.per_row_output_counts = {1, 2};
  ExpectGemmaCppInference(inference_model, {"transcript"}, "topic",
                          std::move(topic_output));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"foo", "bar"}, "transcript"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
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

TEST_F(InferenceModelTest, RunInferenceWithDuplicationAndMultiplePrompts) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Topic for {{transcript}}" }
      }
      inference_task: {
        column_config {
          input_column_names: [ "other_input" ]
          output_column_name: "other_output"
        }
        prompt { prompt_template: "Other for {{other_input}}" }
      }
    }
    gemma_init_config {}
  )pb");
  inference_configuration.gemma_configuration.emplace();

  MockInferenceModel inference_model(inference_configuration);
  // Setup for first inference task
  MockInferenceModel::InferenceOutput topic_output;
  topic_output.tensor = Tensor({"t1", "t2", "t3"}, "topic");
  topic_output.per_row_output_counts = {1, 2};
  ExpectGemmaCppInference(inference_model, {"transcript"}, "topic",
                          std::move(topic_output));

  // Setup for second inference task
  MockInferenceModel::InferenceOutput other_output;
  other_output.tensor = Tensor({"o1", "o2", "o3", "o4"}, "other_output");
  other_output.per_row_output_counts = {1, 2, 1};
  ExpectGemmaCppInference(inference_model, {"other_input"}, "other_output",
                          std::move(other_output));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"foo", "bar"}, "transcript"));
  columns.push_back(Tensor({"a", "b"}, "other_input"));

  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
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

TEST_F(InferenceModelTest,
       RunInferenceMultipleRowsPreservesMetadataAndPrivacyId) {
  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_names: [ "transcript" ]
          output_column_name: "topic"
        }
        prompt { prompt_template: "Topic for {{transcript}}" }
      }
    }
    gemma_init_config {}
  )pb");
  inference_configuration.gemma_configuration.emplace();

  MockInferenceModel inference_model(inference_configuration);
  MockInferenceModel::InferenceOutput topic_output;
  topic_output.tensor = Tensor({"t1", "t2", "t3"}, "topic");
  topic_output.per_row_output_counts = {1, 2};
  ExpectGemmaCppInference(inference_model, {"transcript"}, "topic",
                          std::move(topic_output));

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"foo", "bar"}, "transcript"));

  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), "key_1", "privacy_id_1");
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  EXPECT_EQ(input->GetMetadata(), "key_1");
  ASSERT_TRUE(input->GetPrivacyId().has_value());
  EXPECT_EQ(input->GetPrivacyId().value(), "privacy_id_1");
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
