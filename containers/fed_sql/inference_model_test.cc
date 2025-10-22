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
#include "absl/status/status_matchers.h"
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
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::Prompt;
using ::gcpp::Gemma;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::ByMove;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::UnorderedElementsAre;

class MockInferenceModel : public InferenceModel {
 public:
  MOCK_METHOD(void, BuildGemmaCppModel,
              (const SessionGemmaCppConfiguration& gemma_config), (override));
  MOCK_METHOD(absl::StatusOr<Tensor>, RunGemmaCppInference,
              (const Prompt& prompt,
               (const absl::flat_hash_map<std::string,
                                          absl::Span<const absl::string_view>>&)
                   column_values,
               const std::string& output_column_name),
              (override));
};

TEST(InferenceModelTest, HasModelNone) {
  InferenceModel inference_model = InferenceModel();
  ASSERT_FALSE(inference_model.HasModel());
}

TEST(InferenceModelTest, HasModelGemma) {
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

TEST(InferenceModelTest, HasModelGemmaMultipleInputs) {
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

TEST(InferenceModelTest, BuildModelGemmaValidConfig) {
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

TEST(InferenceModelTest, BuildModelGemmaMissingSessionGemmaCppConfiguration) {
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

TEST(InferenceModelTest, RunInferenceValidConfig) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());
  ON_CALL(inference_model, RunGemmaCppInference)
      .WillByDefault(Invoke(
          [](const Prompt& prompt,
             const absl::flat_hash_map<std::string,
                                       absl::Span<const absl::string_view>>&
                 column_values,
             const std::string& output_column_name) -> absl::StatusOr<Tensor> {
            auto column_value = column_values.at("transcript");
            int64_t tensor_size = column_value.size();
            auto output_data = std::make_unique<MutableStringData>(tensor_size);
            for (int i = 0; i < column_value.size(); ++i) {
              std::string reversed_val(column_value.at(i));
              std::reverse(reversed_val.begin(), reversed_val.end());
              (*output_data)
                  .Add(absl::StrCat(prompt.prompt_template(), "---",
                                    reversed_val));
            }
            return Tensor::Create(DataType::DT_STRING,
                                  TensorShape({tensor_size}),
                                  std::move(output_data), output_column_name);
          }));

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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());

  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  ASSERT_THAT(inference_model.RunInference(columns), IsOk());
  ASSERT_EQ(columns.size(), 1);
  ASSERT_EQ(columns.at(0).name(), "topic");
  ASSERT_EQ(columns.at(0).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(0).AsSpan<absl::string_view>(),
              UnorderedElementsAre("Hello, {{transcript}}---eno",
                                   "Hello, {{transcript}}---owt",
                                   "Hello, {{transcript}}---eerht"));
}

TEST(InferenceModelTest, RunInferenceValidConfigMultipleInputs) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());
  ON_CALL(inference_model, RunGemmaCppInference)
      .WillByDefault(Invoke(
          [](const Prompt& prompt,
             const absl::flat_hash_map<std::string,
                                       absl::Span<const absl::string_view>>&
                 column_values,
             const std::string& output_column_name) -> absl::StatusOr<Tensor> {
            auto transcript_span = column_values.at("transcript");
            auto transcript2_span = column_values.at("transcript2");
            auto num_rows = transcript_span.size();
            auto output_data = std::make_unique<MutableStringData>(num_rows);
            for (int i = 0; i < num_rows; ++i) {
              std::string val1(transcript_span[i]);
              std::string val2(transcript2_span[i]);
              std::reverse(val1.begin(), val1.end());
              std::reverse(val2.begin(), val2.end());
              (*output_data)
                  .Add(absl::StrCat(prompt.prompt_template(), "---", val1, ",",
                                    val2));
            }
            return Tensor::Create(DataType::DT_STRING,
                                  TensorShape({(int64_t)num_rows}),
                                  std::move(output_data), output_column_name);
          }));

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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  std::initializer_list<absl::string_view> transcript2_values = {"aol", "bat",
                                                                 "cat"};
  absl::StatusOr<Tensor> transcript2_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript2_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(
          transcript2_values),
      /*name=*/"transcript2");
  ASSERT_THAT(transcript2_tensor, IsOk());
  columns.push_back(std::move(*transcript2_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  ASSERT_THAT(inference_model.RunInference(columns), IsOk());
  ASSERT_EQ(columns.size(), 1);
  ASSERT_EQ(columns.at(0).name(), "topic");
  ASSERT_EQ(columns.at(0).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(0).AsSpan<absl::string_view>(),
              UnorderedElementsAre(
                  "Hello, {{transcript}}, {{transcript2}}---eno,loa",
                  "Hello, {{transcript}}, {{transcript2}}---owt,tab",
                  "Hello, {{transcript}}, {{transcript2}}---eerht,tac"));
}

TEST(InferenceModelTest, RunInferenceRegexOutput) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());
  ON_CALL(inference_model, RunGemmaCppInference)
      .WillByDefault(Invoke(
          [](const Prompt& prompt,
             const absl::flat_hash_map<std::string,
                                       absl::Span<const absl::string_view>>&
                 column_values,
             const std::string& output_column_name) -> absl::StatusOr<Tensor> {
            auto input_span = column_values.at("transcript");
            auto num_rows = input_span.size();
            auto output_data = std::make_unique<MutableStringData>(num_rows);
            for (int i = 0; i < num_rows; ++i) {
              std::string reversed_val(input_span[i]);
              std::reverse(reversed_val.begin(), reversed_val.end());
              (*output_data).Add(absl::StrCat(reversed_val));
            }
            return Tensor::Create(DataType::DT_STRING,
                                  TensorShape({(int64_t)num_rows}),
                                  std::move(output_data), output_column_name);
          }));

  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_name: "transcript"
          output_column_name: "topic"
        }
        prompt {
          prompt_template: "Outputs Classification: **{reversed transcript}**"
          regex: "Classification:\\s*[*]{1,2}(\\w+)[*]{1,2}"
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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());

  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  ASSERT_THAT(inference_model.RunInference(columns), IsOk());
  ASSERT_EQ(columns.size(), 1);
  ASSERT_EQ(columns.at(0).name(), "topic");
  ASSERT_EQ(columns.at(0).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(0).AsSpan<absl::string_view>(),
              UnorderedElementsAre("eno", "owt", "eerht"));
}

TEST(InferenceModelTest, RunInferenceMultipleInferenceTasks) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());
  ON_CALL(inference_model, RunGemmaCppInference)
      .WillByDefault(Invoke(
          [](const Prompt& prompt,
             const absl::flat_hash_map<std::string,
                                       absl::Span<const absl::string_view>>&
                 column_values,
             const std::string& output_column_name) -> absl::StatusOr<Tensor> {
            absl::Span<const absl::string_view> input_span;
            auto it = column_values.find("transcript");
            if (it != column_values.end()) {
              input_span = it->second;
            } else {
              input_span = column_values.at("input");
            }
            auto num_rows = input_span.size();
            auto output_data = std::make_unique<MutableStringData>(num_rows);
            for (int i = 0; i < num_rows; ++i) {
              std::string reversed_val(input_span[i]);
              std::reverse(reversed_val.begin(), reversed_val.end());
              (*output_data)
                  .Add(absl::StrCat(prompt.prompt_template(), "---",
                                    reversed_val));
            }
            return Tensor::Create(DataType::DT_STRING,
                                  TensorShape({(int64_t)num_rows}),
                                  std::move(output_data), output_column_name);
          }));

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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());

  columns.push_back(std::move(*transcript_tensor));

  std::initializer_list<absl::string_view> input_values = {"uno", "dos",
                                                           "tres"};
  absl::StatusOr<Tensor> input_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(input_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(input_values),
      /*name=*/"input");
  ASSERT_THAT(input_tensor, IsOk());
  columns.push_back(std::move(*input_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  ASSERT_THAT(inference_model.RunInference(columns), IsOk());
  ASSERT_EQ(columns.size(), 2);
  ASSERT_EQ(columns.at(0).name(), "topic");
  ASSERT_EQ(columns.at(0).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(0).AsSpan<absl::string_view>(),
              UnorderedElementsAre("Hello, {{transcript}}---eno",
                                   "Hello, {{transcript}}---owt",
                                   "Hello, {{transcript}}---eerht"));
  ASSERT_EQ(columns.at(1).name(), "output");
  ASSERT_EQ(columns.at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(1).AsSpan<absl::string_view>(),
              UnorderedElementsAre("Good bye, {{input}}---onu",
                                   "Good bye, {{input}}---sod",
                                   "Good bye, {{input}}---sert"));
}

TEST(InferenceModelTest, RunInferenceMultipleInferenceTasksWithRegex) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());
  ON_CALL(inference_model, RunGemmaCppInference)
      .WillByDefault(Invoke(
          [](const Prompt& prompt,
             const absl::flat_hash_map<std::string,
                                       absl::Span<const absl::string_view>>&
                 column_values,
             const std::string& output_column_name) -> absl::StatusOr<Tensor> {
            absl::Span<const absl::string_view> input_span;
            auto it = column_values.find("transcript");
            if (it != column_values.end()) {
              input_span = it->second;
            } else {
              input_span = column_values.at("input");
            }
            auto num_rows = input_span.size();
            auto output_data = std::make_unique<MutableStringData>(num_rows);
            for (int i = 0; i < num_rows; ++i) {
              std::string reversed_val(input_span[i]);
              std::reverse(reversed_val.begin(), reversed_val.end());
              (*output_data).Add(absl::StrCat(reversed_val));
            }
            return Tensor::Create(DataType::DT_STRING,
                                  TensorShape({(int64_t)num_rows}),
                                  std::move(output_data), output_column_name);
          }));

  SessionInferenceConfiguration inference_configuration;
  inference_configuration.initialize_configuration = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      inference_task: {
        column_config {
          input_column_name: "transcript"
          output_column_name: "topic"
        }
        prompt {
          prompt_template: "Outputs Category1: **{reversed transcript}**"
          regex: "Category1:\\s*[*]{1,2}(\\w+)[*]{1,2}"
        }
      }
      inference_task: {
        column_config {
          input_column_name: "input"
          output_column_name: "output"
        }
        prompt {
          prompt_template: "Outputs Category2: ##{reversed transcript}##"
          regex: "Category2:\\s*[#]{1,2}(\\w+)[#]{1,2}"
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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());

  columns.push_back(std::move(*transcript_tensor));

  std::initializer_list<absl::string_view> input_values = {"uno", "dos",
                                                           "tres"};
  absl::StatusOr<Tensor> input_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(input_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(input_values),
      /*name=*/"input");
  ASSERT_THAT(input_tensor, IsOk());
  columns.push_back(std::move(*input_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  ASSERT_THAT(inference_model.RunInference(columns), IsOk());
  ASSERT_EQ(columns.size(), 2);
  ASSERT_EQ(columns.at(0).name(), "topic");
  ASSERT_EQ(columns.at(0).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(0).AsSpan<absl::string_view>(),
              UnorderedElementsAre("eno", "owt", "eerht"));
  ASSERT_EQ(columns.at(1).name(), "output");
  ASSERT_EQ(columns.at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(1).AsSpan<absl::string_view>(),
              UnorderedElementsAre("onu", "sod", "sert"));
}

TEST(InferenceModelTest, RunInferenceKeepsNonPromptColumns) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());
  ON_CALL(inference_model, RunGemmaCppInference)
      .WillByDefault(Invoke(
          [](const Prompt& prompt,
             const absl::flat_hash_map<std::string,
                                       absl::Span<const absl::string_view>>&
                 column_values,
             const std::string& output_column_name) -> absl::StatusOr<Tensor> {
            auto input_span = column_values.at("transcript");
            auto num_rows = input_span.size();
            auto output_data = std::make_unique<MutableStringData>(num_rows);
            for (int i = 0; i < num_rows; ++i) {
              std::string reversed_val(input_span[i]);
              std::reverse(reversed_val.begin(), reversed_val.end());
              (*output_data)
                  .Add(absl::StrCat(prompt.prompt_template(), "---",
                                    reversed_val));
            }
            return Tensor::Create(DataType::DT_STRING,
                                  TensorShape({(int64_t)num_rows}),
                                  std::move(output_data), output_column_name);
          }));

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
  std::initializer_list<absl::string_view> input_str_values = {"uno", "dos",
                                                               "tres"};
  absl::StatusOr<Tensor> input_str_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(input_str_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(input_str_values),
      /*name=*/"input_str_col");
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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  ASSERT_THAT(inference_model.RunInference(columns), IsOk());
  ASSERT_EQ(columns.size(), 3);
  ASSERT_EQ(columns.at(0).name(), "input_str_col");
  ASSERT_EQ(columns.at(0).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(0).AsSpan<absl::string_view>(),
              UnorderedElementsAre("uno", "dos", "tres"));
  ASSERT_EQ(columns.at(1).name(), "input_int_col");
  ASSERT_EQ(columns.at(1).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(1).AsSpan<int64_t>(), UnorderedElementsAre(1, 2, 3));
  ASSERT_EQ(columns.at(2).name(), "topic");
  ASSERT_EQ(columns.at(2).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(2).AsSpan<absl::string_view>(),
              UnorderedElementsAre("Hello, {{transcript}}---eno",
                                   "Hello, {{transcript}}---owt",
                                   "Hello, {{transcript}}---eerht"));
}

TEST(InferenceModelTest, RunInferenceInputColumnNotFound) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());

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
  std::initializer_list<absl::string_view> input_str_values = {"uno", "dos",
                                                               "tres"};
  absl::StatusOr<Tensor> input_str_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(input_str_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(input_str_values),
      /*name=*/"input_str_col");
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

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  auto status = inference_model.RunInference(columns);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      status.message(),
      HasSubstr(
          "Couldn't find an input column transcript to run inference on."));
}

TEST(InferenceModelTest, RunInferenceInputColumnNotFoundMultipleInputs) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());

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
  std::initializer_list<absl::string_view> input_str_values = {"uno", "dos",
                                                               "tres"};
  absl::StatusOr<Tensor> input_str_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(input_str_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(input_str_values),
      /*name=*/"input_str_col");
  ASSERT_THAT(input_str_tensor, IsOk());
  columns.push_back(std::move(*input_str_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  auto status = inference_model.RunInference(columns);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      status.message(),
      HasSubstr(
          "Couldn't find an input column transcript to run inference on."));
}

TEST(InferenceModelTest, RunInferenceNoPrompt) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());

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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  auto status = inference_model.RunInference(columns);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Prompt not found when running inference."));
}

TEST(InferenceModelTest, RunInferenceNoPromptMultipleInputs) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());

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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  auto status = inference_model.RunInference(columns);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Prompt not found when running inference."));
}

TEST(InferenceModelTest, RunInferenceNonStringColumn) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());

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
  std::initializer_list<int64_t> transcript_values = {1, 2, 3};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_INT64,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<int64_t>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  auto status = inference_model.RunInference(columns);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Input column transcript is not of type STRING."));
}

TEST(InferenceModelTest, RunInferenceNonStringColumnMultipleInputs) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());

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
  std::initializer_list<int64_t> transcript_values = {1, 2, 3};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_INT64,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<int64_t>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  auto status = inference_model.RunInference(columns);
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Input column transcript is not of type STRING."));
}

TEST(InferenceModelTest, RunInferenceModelNotInitialized) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());

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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  auto status = inference_model.RunInference(columns);
  ASSERT_EQ(status.code(), absl::StatusCode::kUnimplemented);
  ASSERT_THAT(status.message(),
              HasSubstr("Model must be initialized before running inference."));
}

TEST(InferenceModelTest, RunInferenceModelNotInitializedMultipleInputs) {
  MockInferenceModel inference_model = MockInferenceModel();
  ON_CALL(inference_model, BuildGemmaCppModel).WillByDefault(Return());

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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());
  columns.push_back(std::move(*transcript_tensor));

  auto status = inference_model.RunInference(columns);
  ASSERT_EQ(status.code(), absl::StatusCode::kUnimplemented);
  ASSERT_THAT(status.message(),
              HasSubstr("Model must be initialized before running inference."));
}

TEST(InferenceModelTest, RunInferenceWithRuntimeConfigFlags) {
  MockInferenceModel inference_model;
  ON_CALL(inference_model, BuildGemmaCppModel)
      .WillByDefault(Invoke(
          [&inference_model](const SessionGemmaCppConfiguration& gemma_config) {
            const auto& config =
                inference_model.GetInferenceConfiguration()
                    ->initialize_configuration.inference_config();
            EXPECT_EQ(config.runtime_config().seq_len(), 10000);
          }));
  ON_CALL(inference_model, RunGemmaCppInference)
      .WillByDefault(Invoke(
          [&inference_model](
              const Prompt& prompt,
              const absl::flat_hash_map<std::string,
                                        absl::Span<const absl::string_view>>&
                  column_values,
              const std::string& output_column_name) -> absl::StatusOr<Tensor> {
            const auto& config =
                inference_model.GetInferenceConfiguration()
                    ->initialize_configuration.inference_config();
            EXPECT_EQ(config.runtime_config().max_prompt_size(), 100);
            EXPECT_EQ(config.runtime_config().max_generated_tokens(), 50);
            EXPECT_EQ(config.runtime_config().temperature_diff(), -0.5);
            auto column_value = column_values.at("transcript");
            int64_t tensor_size = column_value.size();
            auto output_data = std::make_unique<MutableStringData>(tensor_size);
            for (int i = 0; i < column_value.size(); ++i) {
              std::string reversed_val(column_value.at(i));
              std::reverse(reversed_val.begin(), reversed_val.end());
              (*output_data)
                  .Add(absl::StrCat(prompt.prompt_template(), "---",
                                    reversed_val));
            }
            return Tensor::Create(DataType::DT_STRING,
                                  TensorShape({tensor_size}),
                                  std::move(output_data), output_column_name);
          }));

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
  std::initializer_list<absl::string_view> transcript_values = {"one", "two",
                                                                "three"};
  absl::StatusOr<Tensor> transcript_tensor = Tensor::Create(
      DataType::DT_STRING,
      TensorShape({static_cast<int64_t>(transcript_values.size())}),
      std::make_unique<MutableVectorData<absl::string_view>>(transcript_values),
      /*name=*/"transcript");
  ASSERT_THAT(transcript_tensor, IsOk());

  columns.push_back(std::move(*transcript_tensor));

  ASSERT_THAT(inference_model.BuildModel(inference_configuration), IsOk());
  ASSERT_THAT(inference_model.RunInference(columns), IsOk());
  ASSERT_EQ(columns.size(), 1);
  ASSERT_EQ(columns.at(0).name(), "topic");
  ASSERT_EQ(columns.at(0).shape().dim_sizes()[0], 3);
  EXPECT_THAT(columns.at(0).AsSpan<absl::string_view>(),
              UnorderedElementsAre("Hello, {{transcript}}---eno",
                                   "Hello, {{transcript}}---owt",
                                   "Hello, {{transcript}}---eerht"));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
