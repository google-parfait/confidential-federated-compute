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

#include "containers/fed_sql/inference_model_helper.h"

#include <string>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "containers/sql/input.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::confidential_federated_compute::sql::Input;
using ::confidential_federated_compute::sql::RowView;
using ::fcp::confidentialcompute::Prompt;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class InferenceModelInternalTest : public ::testing::Test {
 protected:
  // Helper to build a string tensor for our test data.
  absl::StatusOr<Tensor> CreateStringTensor(
      std::initializer_list<absl::string_view> values,
      const std::string& name) {
    return Tensor::Create(
        DataType::DT_STRING, TensorShape({static_cast<int64_t>(values.size())}),
        std::make_unique<MutableVectorData<absl::string_view>>(values), name);
  }
};

TEST_F(InferenceModelInternalTest, PopulatePromptTemplateSuccess) {
  Prompt prompt;
  prompt.set_prompt_template("User: {user}, Message: {msg}, Session: {id}");

  absl::StatusOr<Tensor> user_tensor = CreateStringTensor({"John"}, "user");
  ASSERT_THAT(user_tensor, IsOk());
  absl::StatusOr<Tensor> msg_tensor =
      CreateStringTensor({"Are you okay?"}, "msg");
  ASSERT_THAT(msg_tensor, IsOk());
  absl::StatusOr<Tensor> id_tensor = CreateStringTensor({"42"}, "id");
  ASSERT_THAT(id_tensor, IsOk());

  std::vector<Tensor> columns;
  columns.push_back(std::move(*user_tensor));
  columns.push_back(std::move(*msg_tensor));
  columns.push_back(std::move(*id_tensor));

  fcp::confidentialcompute::BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
  ASSERT_THAT(input, IsOk());
  absl::StatusOr<RowView> row = input->GetRow(0);
  ASSERT_THAT(row, IsOk());

  const std::string column_names[] = {"user", "msg", "id"};
  const size_t indices[] = {0, 1, 2};
  const std::string output_column_name = "output";
  const size_t max_prompt_size = 1000;

  InferencePromptProcessor prompt_processor;

  std::string expected_prompt =
      "User: John, Message: Are you okay?, Session: 42";
  EXPECT_THAT(prompt_processor.PopulatePromptTemplate(
                  prompt, *row, column_names, indices, output_column_name,
                  max_prompt_size),
              IsOkAndHolds(expected_prompt));
}

TEST_F(InferenceModelInternalTest,
       PopulatePromptTemplateWithSystemInstructions) {
  Prompt prompt;
  prompt.set_prompt_template("User: {user}");
  prompt.set_parser(Prompt::PARSER_AUTO);

  absl::StatusOr<Tensor> user_tensor = CreateStringTensor({"John"}, "user");
  ASSERT_THAT(user_tensor, IsOk());
  std::vector<Tensor> columns;
  columns.push_back(std::move(*user_tensor));
  fcp::confidentialcompute::BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
  ASSERT_THAT(input, IsOk());
  absl::StatusOr<RowView> row = input->GetRow(0);
  ASSERT_THAT(row, IsOk());

  const std::string column_names[] = {"user"};
  const size_t indices[] = {0};
  const std::string output_column_name = "output";
  const size_t max_prompt_size = 1000;
  InferencePromptProcessor prompt_processor;

  std::string expected_prompt = "User: John";
  absl::StrAppend(
      &expected_prompt, "\n***System Instruction***\n",
      "You must respond with a valid JSON object. The key of the JSON object "
      "must be '",
      output_column_name,
      "' and its value must be a JSON array. Do not include any other text or "
      "explanation outside of the JSON object.\n",
      "Example format:\n", "```json\n", "{\n", "  \"", output_column_name,
      "\": [\"", output_column_name, "_val_0\", \"", output_column_name,
      "_val_1\", \"", output_column_name, "_val_2\" ...]\n", "}\n", "```");

  EXPECT_THAT(prompt_processor.PopulatePromptTemplate(
                  prompt, *row, column_names, indices, output_column_name,
                  max_prompt_size),
              IsOkAndHolds(expected_prompt));
}

TEST_F(InferenceModelInternalTest, PopulatePromptTemplateTruncation) {
  Prompt prompt;
  prompt.set_prompt_template("User: {user}");

  absl::StatusOr<Tensor> user_tensor = CreateStringTensor(
      {"This is a very long username that will cause the prompt to exceed the "
       "max size"},
      "user");
  ASSERT_THAT(user_tensor, IsOk());
  std::vector<Tensor> columns;
  columns.push_back(std::move(*user_tensor));
  fcp::confidentialcompute::BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
  ASSERT_THAT(input, IsOk());
  absl::StatusOr<RowView> row = input->GetRow(0);
  ASSERT_THAT(row, IsOk());

  const std::string column_names[] = {"user"};
  const size_t indices[] = {0};
  const std::string output_column_name = "output";
  const size_t max_prompt_size = 20;
  InferencePromptProcessor prompt_processor;

  std::string expected_prompt = "User: This is a very";  // Truncated to 20
  EXPECT_THAT(prompt_processor.PopulatePromptTemplate(
                  prompt, *row, column_names, indices, output_column_name,
                  max_prompt_size),
              IsOkAndHolds(expected_prompt));
}

TEST_F(InferenceModelInternalTest, ProcessInferenceOutputNoAutoParser) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  std::string output_string = R"({"topic": ["foo", "bar"]})";
  const std::string expected_output = output_string;
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_THAT(result, IsOkAndHolds(1));
  const auto* data_ptr =
      static_cast<const absl::string_view*>(output_string_data->data());
  EXPECT_THAT(absl::MakeSpan(data_ptr, *result), ElementsAre(expected_output));
}

TEST_F(InferenceModelInternalTest, ProcessInferenceOutputSuccess) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_AUTO);
  std::string output_string = R"({"topic": ["foo", "bar"]})";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_THAT(result, IsOkAndHolds(2));
  const auto* data_ptr =
      static_cast<const absl::string_view*>(output_string_data->data());
  EXPECT_THAT(absl::MakeSpan(data_ptr, *result), ElementsAre("foo", "bar"));
}

TEST_F(InferenceModelInternalTest, ProcessInferenceOutputWithMarkdown) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_AUTO);
  std::string output_string =
      "Here is the result:\n```json\n{\"topic\": [\"foo\"]}\n```";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_THAT(result, IsOkAndHolds(1));
  const auto* data_ptr =
      static_cast<const absl::string_view*>(output_string_data->data());
  EXPECT_THAT(absl::MakeSpan(data_ptr, *result), ElementsAre("foo"));
}

TEST_F(InferenceModelInternalTest, ProcessInferenceOutputInvalidJson) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_AUTO);
  std::string output_string = R"({"topic": ["foo", "bar"]invalid})";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              HasSubstr("Failed to parse model output as JSON"));
}

TEST_F(InferenceModelInternalTest, ProcessInferenceOutputNotAnObject) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_AUTO);
  std::string output_string = R"(["foo", "bar"])";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              HasSubstr("Failed to parse model output as JSON"));
}

TEST_F(InferenceModelInternalTest, ProcessInferenceOutputMissingKey) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_AUTO);
  std::string output_string = R"({"wrong_key": ["foo", "bar"]})";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              HasSubstr("Could not find key 'topic' in JSON output."));
}

TEST_F(InferenceModelInternalTest, ProcessInferenceOutputValueNotArray) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_AUTO);
  std::string output_string = R"({"topic": "foo"})";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              HasSubstr("Value for key 'topic' is not a JSON array."));
}

TEST_F(InferenceModelInternalTest, ProcessInferenceOutputArrayWithMixedTypes) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_AUTO);
  std::string output_string =
      R"({"topic": ["foo", 123, 45.6, true, false, -7]})";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_THAT(result, IsOkAndHolds(6));
  const auto* data_ptr =
      static_cast<const absl::string_view*>(output_string_data->data());
  EXPECT_THAT(absl::MakeSpan(data_ptr, *result),
              ElementsAre("foo", "123", "45.6", "true", "false", "-7"));
}

TEST_F(InferenceModelInternalTest,
       ProcessInferenceOutputArrayWithUnsupportedType) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_AUTO);
  std::string output_string = R"({"topic": ["foo", null]})";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              HasSubstr("JSON array contains an unsupported element of type"));
}

TEST_F(InferenceModelInternalTest, ProcessInferenceOutputWithRegex) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_AUTO);
  prompt.set_regex("Result: <([^>]+)>");
  std::string output_string =
      R"({"topic": ["Result: <foo>", "Result: <bar>"]})";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_THAT(result, IsOkAndHolds(2));
  const auto* data_ptr =
      static_cast<const absl::string_view*>(output_string_data->data());
  EXPECT_THAT(absl::MakeSpan(data_ptr, *result), ElementsAre("foo", "bar"));
}

TEST_F(InferenceModelInternalTest,
       ProcessInferenceOutputWithDelimiterParserDefault) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_DELIMITER);
  std::string output_string = "foo,bar,baz";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_THAT(result, IsOkAndHolds(3));
  const auto* data_ptr =
      static_cast<const absl::string_view*>(output_string_data->data());
  EXPECT_THAT(absl::MakeSpan(data_ptr, *result),
              ElementsAre("foo", "bar", "baz"));
}

TEST_F(InferenceModelInternalTest,
       ProcessInferenceOutputWithDelimiterParserCustom) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_DELIMITER);
  prompt.mutable_delimiter_parser()->set_delimiter("||");
  std::string output_string = "foo||bar||baz";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_THAT(result, IsOkAndHolds(3));
  const auto* data_ptr =
      static_cast<const absl::string_view*>(output_string_data->data());
  EXPECT_THAT(absl::MakeSpan(data_ptr, *result),
              ElementsAre("foo", "bar", "baz"));
}

TEST_F(InferenceModelInternalTest,
       ProcessInferenceOutputWithDelimiterParserSingleValue) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_DELIMITER);
  std::string output_string = "foo";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_THAT(result, IsOkAndHolds(1));
  const auto* data_ptr =
      static_cast<const absl::string_view*>(output_string_data->data());
  EXPECT_THAT(absl::MakeSpan(data_ptr, *result), ElementsAre("foo"));
}

TEST_F(InferenceModelInternalTest,
       ProcessInferenceOutputWithDelimiterParserAndRegex) {
  InferenceOutputProcessor processor;
  Prompt prompt;
  prompt.set_parser(Prompt::PARSER_DELIMITER);
  prompt.set_regex("val:([a-z]+)");
  std::string output_string = "val:foo,val:bar";
  std::string output_column_name = "topic";
  auto output_string_data = std::make_unique<MutableStringData>(0);
  auto result = processor.ProcessInferenceOutput(
      prompt, std::move(output_string), output_column_name,
      output_string_data.get());
  ASSERT_THAT(result, IsOkAndHolds(2));
  const auto* data_ptr =
      static_cast<const absl::string_view*>(output_string_data->data());
  EXPECT_THAT(absl::MakeSpan(data_ptr, *result), ElementsAre("foo", "bar"));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql