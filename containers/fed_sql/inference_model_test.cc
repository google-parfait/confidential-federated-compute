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
#include "containers/fed_sql/testing/mocks.h"
#include "containers/sql/input.h"
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
using ::confidential_federated_compute::fed_sql::testing::MockInferenceModel;
using ::confidential_federated_compute::sql::Input;
using ::fcp::confidentialcompute::BlobHeader;
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
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::UnorderedElementsAre;

class InferenceModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ON_CALL(inference_model_, BuildGemmaCppModel).WillByDefault(Return());
  }

  // Helper to build a string tensor for our test data.
  absl::StatusOr<Tensor> CreateStringTensor(
      std::initializer_list<absl::string_view> values,
      const std::string& name) {
    return Tensor::Create(
        DataType::DT_STRING, TensorShape({static_cast<int64_t>(values.size())}),
        std::make_unique<MutableVectorData<absl::string_view>>(values), name);
  }

  // Helper to set expectations on RunGemmaCppInference.
  void ExpectInference(const std::vector<std::string>& input_names,
                       const std::string& output_name, Tensor output_tensor) {
    EXPECT_CALL(inference_model_,
                RunGemmaCppInference(_, _, _, Eq(output_name)))
        .WillOnce(Return(ByMove(std::move(output_tensor))));
  }

  MockInferenceModel inference_model_;
};

TEST_F(InferenceModelTest, HasModelNone) {
  InferenceModel inference_model = InferenceModel();
  ASSERT_FALSE(inference_model.HasModel());
}

TEST_F(InferenceModelTest, HasModelGemma) {
  MockInferenceModel inference_model = MockInferenceModel();
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

  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());
  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  ASSERT_TRUE(inference_model.HasModel());
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

  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());
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
          input_column_name: "transcript"
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

  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());
  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
}

TEST_F(InferenceModelTest, BuildModelGemmaMissingSessionGemmaCppConfiguration) {
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
    }
    gemma_init_config {
      tokenizer_configuration_id: "tokenizer_configuration_id"
    }
  )pb");

  auto status = inference_model.BuildModel(inference_configuration);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Missing session Gemma configuration"));
}

TEST_F(InferenceModelTest, RunInferenceValidConfig) {
  absl::StatusOr<Tensor> output_tensor =
      CreateStringTensor({"1", "2", "3"}, "topic");
  ASSERT_THAT(output_tensor, IsOk());
  ExpectInference({"transcript"}, "topic", std::move(*output_tensor));

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
  absl::StatusOr<Tensor> transcript_tensor =
      CreateStringTensor({"one", "two", "three"}, "transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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

TEST_F(InferenceModelTest, RunInferenceValidConfigMultipleInputs) {
  absl::StatusOr<Tensor> output_tensor =
      CreateStringTensor({"1", "2", "3"}, "topic");
  ASSERT_THAT(output_tensor, IsOk());
  ExpectInference({"transcript", "transcript2"}, "topic",
                  std::move(*output_tensor));

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
  absl::StatusOr<Tensor> transcript_tensor =
      CreateStringTensor({"one", "two", "three"}, "transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  absl::StatusOr<Tensor> transcript2_tensor =
      CreateStringTensor({"aol", "bat", "cat"}, "transcript2");
  ASSERT_THAT(transcript2_tensor, IsOk());
  columns.push_back(std::move(*transcript2_tensor));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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
  absl::StatusOr<Tensor> output_tensor =
      CreateStringTensor({"1", "2", "3"}, "topic");
  ASSERT_THAT(output_tensor, IsOk());
  ExpectInference({"transcript"}, "topic", *std::move(output_tensor));
  output_tensor = CreateStringTensor({"1", "2", "3"}, "output");
  ASSERT_THAT(output_tensor, IsOk());
  ExpectInference({"input"}, "output", *std::move(output_tensor));

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
      inference_task: {
        column_config {
          input_column_name: "input"
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
  absl::StatusOr<Tensor> transcript_tensor =
      CreateStringTensor({"one", "two", "three"}, "transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  absl::StatusOr<Tensor> input_tensor =
      CreateStringTensor({"uno", "dos", "tres"}, "input");
  ASSERT_THAT(input_tensor, IsOk());
  columns.push_back(std::move(*input_tensor));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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
  absl::StatusOr<Tensor> output_tensor =
      CreateStringTensor({"1", "2", "3"}, "topic");
  ASSERT_THAT(output_tensor, IsOk());
  ExpectInference({"transcript"}, "topic", *std::move(output_tensor));

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
  absl::StatusOr<Tensor> input_str_tensor =
      CreateStringTensor({"uno", "dos", "tres"}, "input_str_col");
  ASSERT_THAT(input_str_tensor, IsOk());
  columns.push_back(std::move(*input_str_tensor));

  // Non-prompt int column.
  std::initializer_list<int64_t> input_int_values = {1, 2, 3};
  absl::StatusOr<Tensor> input_int_tensor = Tensor::Create(
      DataType::DT_INT64,
      TensorShape({static_cast<int64_t>(input_int_values.size())}),
      std::make_unique<MutableVectorData<int64_t>>(input_int_values),
      /*name=*/"input_int_col");
  ASSERT_THAT(input_int_tensor, IsOk());
  columns.push_back(std::move(*input_int_tensor));

  // Prompt column.
  absl::StatusOr<Tensor> transcript_tensor =
      CreateStringTensor({"one", "two", "three"}, "transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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
  EXPECT_THAT(output_tensors->at(1).AsSpan<int64_t>(),
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
          input_column_name: "transcript"
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
  absl::StatusOr<Tensor> input_str_tensor =
      CreateStringTensor({"uno", "dos", "tres"}, "input_str_col");
  ASSERT_THAT(input_str_tensor, IsOk());
  columns.push_back(std::move(*input_str_tensor));

  std::initializer_list<int64_t> input_int_values = {1, 2, 3};
  absl::StatusOr<Tensor> input_int_tensor = Tensor::Create(
      DataType::DT_INT64,
      TensorShape({static_cast<int64_t>(input_int_values.size())}),
      std::make_unique<MutableVectorData<int64_t>>(input_int_values),
      /*name*/ "input_int_col");
  ASSERT_THAT(input_int_tensor, IsOk());

  columns.push_back(std::move(*input_int_tensor));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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
  absl::StatusOr<Tensor> input_str_tensor =
      CreateStringTensor({"uno", "dos", "tres"}, "input_str_col");
  ASSERT_THAT(input_str_tensor, IsOk());
  columns.push_back(std::move(*input_str_tensor));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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
          input_column_name: "transcript"
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
  absl::StatusOr<Tensor> transcript_tensor =
      CreateStringTensor({"one", "two", "three"}, "transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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
  absl::StatusOr<Tensor> transcript_tensor =
      CreateStringTensor({"one", "two", "three"}, "transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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
          input_column_name: "transcript"
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
  absl::StatusOr<Tensor> transcript_tensor =
      CreateStringTensor({"one", "two", "three"}, "transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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
  absl::StatusOr<Tensor> transcript_tensor =
      CreateStringTensor({"one", "two", "three"}, "transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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
          }));
  absl::StatusOr<Tensor> output_tensor =
      CreateStringTensor({"r1", "r2", "r3"}, "topic");
  ASSERT_THAT(output_tensor, IsOk());
  EXPECT_CALL(inference_model_, RunGemmaCppInference(_, _, _, _))
      .WillOnce([this, output_tensor = std::move(*output_tensor)](
                    const Prompt&, const Input&, absl::Span<const size_t>,
                    const std::string&) mutable -> absl::StatusOr<Tensor> {
        const auto& config = inference_model_.GetInferenceConfiguration()
                                 ->initialize_configuration.inference_config();
        EXPECT_EQ(config.runtime_config().max_prompt_size(), 100);
        EXPECT_EQ(config.runtime_config().max_generated_tokens(), 50);
        EXPECT_EQ(config.runtime_config().temperature_diff(), -0.5);
        return std::move(output_tensor);
      });

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
  absl::StatusOr<Tensor> transcript_tensor =
      CreateStringTensor({"one", "two", "three"}, "transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model_.BuildModel(inference_configuration), IsOk());
  BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
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

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
