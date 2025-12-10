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

#include <regex>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "containers/sql/input.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "google/protobuf/struct.pb.h"
#include "google/protobuf/util/json_util.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::confidential_federated_compute::sql::RowView;
using ::fcp::confidentialcompute::Prompt;
using ::google::protobuf::Struct;
using ::google::protobuf::Value;
using ::google::protobuf::util::JsonParseOptions;
using ::google::protobuf::util::JsonStringToMessage;
using ::tensorflow_federated::aggregation::DataType;

}  // namespace

InferenceOutputProcessor::InferenceOutputProcessor() {}

std::string InferenceOutputProcessor::RegexMatch(const std::string& text,
                                                 const std::regex& regex) {
  std::smatch match;
  if (std::regex_match(text, match, regex) && match.size() > 1) {
    return match[1];
  }
  return text;
}

absl::StatusOr<size_t> InferenceOutputProcessor::ProcessInferenceOutput(
    const Prompt& prompt, std::string&& inference_output,
    const std::string& output_column_name,
    tensorflow_federated::aggregation::MutableStringData* output_string_data) {
  std::optional<std::regex> regex;
  if (!prompt.regex().empty()) {
    try {
      regex.emplace(prompt.regex());
    } catch (const std::regex_error& e) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid regex provided in prompt: '", prompt.regex(),
                       "': ", e.what()));
    }
  }

  if (prompt.parser() != Prompt::PARSER_AUTO) {
    std::string value = std::move(inference_output);
    if (regex.has_value()) {
      value = RegexMatch(value, *regex);
    }
    output_string_data->Add(std::move(value));
    return 1;
  }

  std::string json_string = std::move(inference_output);
  std::regex json_block_regex("```json\\s*\\n?([\\s\\S]*?)\\n?```");
  std::smatch match;
  if (std::regex_search(json_string, match, json_block_regex) &&
      match.size() > 1) {
    json_string = match[1].str();
  }

  Struct json_struct;
  JsonParseOptions options;
  options.ignore_unknown_fields = true;
  absl::Status status = JsonStringToMessage(json_string, &json_struct, options);

  if (!status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse model output as JSON: ", status.message()));
  }

  const auto& fields = json_struct.fields();
  auto it = fields.find(output_column_name);
  if (it == fields.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Could not find key '", output_column_name, "' in JSON output."));
  }

  const auto& value = it->second;
  if (value.kind_case() != Value::kListValue) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Value for key '", output_column_name, "' is not a JSON array."));
  }

  size_t num_values_added = 0;
  for (const auto& element : value.list_value().values()) {
    if (element.kind_case() != Value::kStringValue) {
      return absl::InvalidArgumentError(
          "JSON array contains non-string elements.");
    }
    if (regex.has_value()) {
      output_string_data->Add(RegexMatch(element.string_value(), *regex));
    } else {
      output_string_data->Add(std::string(element.string_value()));
    }
    num_values_added++;
  }
  return num_values_added;
}

InferencePromptProcessor::InferencePromptProcessor() {}

void InferencePromptProcessor::AppendSystemInstructions(
    std::string& prompt, const std::string& output_column_name) {
  // Modifies a prompt to include instructions for the model to generate
  // output in a JSON format that can be parsed automatically.
  absl::StrAppend(
      &prompt, "\n***System Instruction***\n",
      "You must respond with a valid JSON object. The key of the JSON "
      "object "
      "must be '",
      output_column_name,
      "' and its value must be a JSON array. Do not include any other text "
      "or "
      "explanation outside of the JSON object.\n",
      "Example format:\n", "```json\n", "{\n", "  \"", output_column_name,
      "\": [\"", output_column_name, "_val_0\", \"", output_column_name,
      "_val_1\", \"", output_column_name, "_val_2\" ...]\n", "}\n", "```");
}

absl::StatusOr<std::string> InferencePromptProcessor::PopulatePromptTemplate(
    const Prompt& prompt, const RowView& row,
    absl::Span<const std::string> column_names,
    absl::Span<const size_t> input_column_indices,
    const std::string& output_column_name, size_t max_prompt_size) {
  std::string populated_prompt(prompt.prompt_template());
  for (size_t input_column_index : input_column_indices) {
    if (input_column_index >= column_names.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input column index ", input_column_index, " is out of bounds."));
    }
    if (row.GetColumnType(input_column_index) != DataType::DT_STRING) {
      return absl::InvalidArgumentError(
          "Only string input columns are supported for inference.");
    }
    absl::string_view column_value =
        row.GetValue<absl::string_view>(input_column_index);
    const std::string placeholder =
        absl::StrCat("{", column_names[input_column_index], "}");
    size_t start_position = 0;
    while ((start_position = populated_prompt.find(
                placeholder, start_position)) != std::string::npos) {
      populated_prompt.replace(start_position, placeholder.length(),
                               column_value);
      start_position += column_value.length();  // Move past the replacement
    }
  }
  if (populated_prompt.size() > max_prompt_size) {
    populated_prompt.resize(max_prompt_size);
  }
  if (prompt.parser() == Prompt::PARSER_AUTO) {
    AppendSystemInstructions(populated_prompt, output_column_name);
  }
  return populated_prompt;
}

}  // namespace confidential_federated_compute::fed_sql
