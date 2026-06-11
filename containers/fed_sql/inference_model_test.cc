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
#include "containers/common/input.h"
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
  void SetUp() override {
    ON_CALL(inference_model_, BuildGemmaCppModel)
        .WillByDefault(Return(absl::OkStatus()));
    ON_CALL(inference_model_, BuildLlamaCppModel)
        .WillByDefault(Return(absl::OkStatus()));
  }

  // Helper to set expectations on RunGemmaCppInference.
  void ExpectGemmaCppInference(
      const std::vector<std::string>& input_names,
      const std::string& output_name,
      absl::StatusOr<MockInferenceModel::InferenceOutput> output) {
    EXPECT_CALL(inference_model_,
                RunGemmaCppInference(_, _, _, Eq(output_name)))
        .WillOnce(Invoke([output = std::move(output)]() mutable {
          return std::move(output);
        }));
  }

  MockInferenceModel inference_model_;
};

TEST_F(InferenceModelTest, RunGemmaCppInferenceParserAuto) {
  MockInferenceModel::InferenceOutput output;
  output.tensor = Tensor({"1", "2", "3"}, "topic");
  output.per_row_output_counts = {1, 1, 1};
  EXPECT_CALL(
      inference_model_,
      RunGemmaCppInference(Property(&Prompt::parser, Eq(Prompt::PARSER_AUTO)),
                           _, _, Eq("topic")))
      .WillOnce(Invoke([output = std::move(output)]() mutable {
        return std::move(output);
      }));

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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
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
  EXPECT_CALL(inference_model_, RunGemmaCppInference(_, _, _, _))
      .WillOnce(
          Invoke([]() { return absl::InvalidArgumentError("Bad JSON"); }));

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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one"}, "transcript"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model_.RunInference(*input);
  EXPECT_THAT(status, Not(IsOk()));
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.message(), HasSubstr("Bad JSON"));
}

TEST_F(InferenceModelTest, HasModelNone) {
  InferenceModel inference_model = InferenceModel();
  ASSERT_FALSE(inference_model.HasModel());
}

TEST_F(InferenceModelTest, HasModelGemmaMultipleInputs) {
  MockInferenceModel inference_model = MockInferenceModel();
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

  ON_CALL(inference_model, BuildGemmaCppModel)
      .WillByDefault(Return(absl::OkStatus()));
  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  ASSERT_TRUE(inference_model.HasModel());
}

TEST_F(InferenceModelTest, BuildModelGemmaValidConfig) {
  MockInferenceModel inference_model = MockInferenceModel();
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

  ON_CALL(inference_model, BuildGemmaCppModel)
      .WillByDefault(Return(absl::OkStatus()));
  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
}

TEST_F(InferenceModelTest, BuildModelLlamaValidConfig) {
  MockInferenceModel inference_model = MockInferenceModel();
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

  ON_CALL(inference_model, BuildLlamaCppModel)
      .WillByDefault(Return(absl::OkStatus()));
  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
}

TEST_F(InferenceModelTest, BuildModelGemmaMissingSessionGemmaCppConfiguration) {
  InferenceModel inference_model = InferenceModel();
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

  auto status = inference_model.BuildModel(inference_configuration);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  // gemma_configuration is missing from inference_configuration.
  ASSERT_THAT(status.message(),
              HasSubstr("Missing session gemma.cpp configuration"));
}

TEST_F(InferenceModelTest, BuildModelLlamaMissingSessionLlamaCppConfiguration) {
  InferenceModel inference_model = InferenceModel();
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

  auto status = inference_model.BuildModel(inference_configuration);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  // llama_configuration is missing from inference_configuration.
  ASSERT_THAT(status.message(),
              HasSubstr("Missing session llama.cpp configuration"));
}

TEST_F(InferenceModelTest, RunGemmaCppInferenceValidConfig) {
  MockInferenceModel::InferenceOutput output;
  output.tensor = Tensor({"1", "2", "3"}, "topic");
  output.per_row_output_counts = {1, 1, 1};
  ExpectGemmaCppInference({"transcript"}, "topic", std::move(output));

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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
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
  MockInferenceModel::InferenceOutput output;
  output.tensor = Tensor({"1", "2", "3"}, "topic");
  output.per_row_output_counts = {1, 1, 1};
  ExpectGemmaCppInference({"transcript", "transcript2"}, "topic",
                          std::move(output));

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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));
  columns.push_back(Tensor({"aol", "bat", "cat"}, "transcript2"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
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
  MockInferenceModel::InferenceOutput topic_output;
  topic_output.tensor = Tensor({"1", "2", "3"}, "topic");
  ;
  topic_output.per_row_output_counts = {1, 1, 1};
  ExpectGemmaCppInference({"transcript"}, "topic", std::move(topic_output));

  MockInferenceModel::InferenceOutput output_output;
  output_output.tensor = Tensor({"1", "2", "3"}, "output");
  output_output.per_row_output_counts = {1, 1, 1};
  ExpectGemmaCppInference({"input"}, "output", std::move(output_output));

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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
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
  MockInferenceModel::InferenceOutput output;
  output.tensor = Tensor({"1", "2", "3"}, "topic");
  output.per_row_output_counts = {1, 1, 1};
  ExpectGemmaCppInference({"transcript"}, "topic", std::move(output));

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

  std::vector<Tensor> columns;
  // Non-prompt string column.
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input_str_col"));

  // Non-prompt int column.
  columns.push_back(Tensor({1, 2, 3}, "input_int_col"));

  // Prompt column.
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input_str_col"));
  columns.push_back(Tensor({1, 2, 3}, "input_int_col"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model_.RunInference(*input);
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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"uno", "dos", "tres"}, "input_str_col"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model_.RunInference(*input);
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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model_.RunInference(*input);
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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model_.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Prompt not found when running inference."));
}

TEST_F(InferenceModelTest, RunInferenceModelNotInitialized) {
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

  std::vector<Tensor> columns;
  // Prompt column.
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model_.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kUnimplemented);
  ASSERT_THAT(status.message(),
              HasSubstr("Model must be initialized before running inference."));
}

TEST_F(InferenceModelTest, RunInferenceModelNotInitializedMultipleInputs) {
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

  std::vector<Tensor> columns;
  // Prompt column.
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  auto status = inference_model_.RunInference(*input);
  ASSERT_EQ(status.code(), absl::StatusCode::kUnimplemented);
  ASSERT_THAT(status.message(),
              HasSubstr("Model must be initialized before running inference."));
}

TEST_F(InferenceModelTest, RunInferenceWithRuntimeConfigFlags) {
  ON_CALL(inference_model_, BuildGemmaCppModel)
      .WillByDefault(
          Invoke([this](const SessionGemmaCppConfiguration& gemma_config) {
            const auto& config =
                inference_model_.GetInferenceConfiguration()
                    ->initialize_configuration.inference_config();
            EXPECT_EQ(config.runtime_config().seq_len(), 10000);
            return absl::OkStatus();
          }));
  EXPECT_CALL(inference_model_, RunGemmaCppInference(_, _, _, _))
      .WillOnce([this](const Prompt&, const Input&, absl::Span<const size_t>,
                       const std::string&) mutable
                    -> absl::StatusOr<MockInferenceModel::InferenceOutput> {
        const auto& config = inference_model_.GetInferenceConfiguration()
                                 ->initialize_configuration.inference_config();
        EXPECT_EQ(config.runtime_config().max_prompt_size(), 100);
        EXPECT_EQ(config.runtime_config().max_generated_tokens(), 50);
        EXPECT_EQ(config.runtime_config().temperature_diff(), -0.5);
        MockInferenceModel::InferenceOutput output;
        output.tensor = Tensor({"r1", "r2", "r3"}, "topic");
        output.per_row_output_counts = {1, 1, 1};
        return std::move(output);
      });

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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
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
  EXPECT_CALL(inference_model_,
              RunLlamaCppInference(_, _, _, Eq(output_column_name)))
      .WillOnce(Invoke([output = std::move(output)]() mutable {
        return std::move(output);
      }));
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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript_1"));
  columns.push_back(Tensor({"aol", "bat", "cat"}, "transcript_2"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
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
  EXPECT_CALL(inference_model_,
              RunLlamaCppInference(_, _, _, Eq(output_column_name)))
      .WillOnce(Invoke([output = std::move(output)]() mutable {
        return std::move(output);
      }));

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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"one", "two", "three"}, "transcript"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
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
  MockInferenceModel::InferenceOutput topic_output;
  topic_output.tensor = Tensor({"t1", "t2", "t3"}, "topic");
  topic_output.per_row_output_counts = {1, 2};
  ExpectGemmaCppInference({"transcript"}, "topic", std::move(topic_output));

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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"foo", "bar"}, "transcript"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
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
  // Setup for first inference task
  MockInferenceModel::InferenceOutput topic_output;
  topic_output.tensor = Tensor({"t1", "t2", "t3"}, "topic");
  topic_output.per_row_output_counts = {1, 2};
  ExpectGemmaCppInference({"transcript"}, "topic", std::move(topic_output));

  // Setup for second inference task
  MockInferenceModel::InferenceOutput other_output;
  other_output.tensor = Tensor({"o1", "o2", "o3", "o4"}, "other_output");
  other_output.per_row_output_counts = {1, 2, 1};
  ExpectGemmaCppInference({"other_input"}, "other_output",
                          std::move(other_output));

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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"foo", "bar"}, "transcript"));
  columns.push_back(Tensor({"a", "b"}, "other_input"));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  absl::StatusOr<Input> input = Input::CreateFromTensors(std::move(columns));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
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
  MockInferenceModel::InferenceOutput topic_output;
  topic_output.tensor = Tensor({"t1", "t2", "t3"}, "topic");
  topic_output.per_row_output_counts = {1, 2};
  ExpectGemmaCppInference({"transcript"}, "topic", std::move(topic_output));
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

  std::vector<Tensor> columns;
  columns.push_back(Tensor({"foo", "bar"}, "transcript"));
  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());

  Tensor privacy_id_tensor("privacy_id_1", "privacy_id");

  absl::StatusOr<Input> input = Input::CreateFromTensors(
      std::move(columns), "key_1", std::move(privacy_id_tensor));
  ASSERT_THAT(input, IsOk());
  ASSERT_THAT(inference_model_.RunInference(*input), IsOk());
  absl::StatusOr<std::vector<Tensor>> output_tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(output_tensors, IsOk());
  EXPECT_EQ(input->GetMetadata(), "key_1");
  ASSERT_TRUE(input->GetPrivacyId().has_value());
  EXPECT_EQ(input->GetPrivacyId().value(), "privacy_id_1");
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
