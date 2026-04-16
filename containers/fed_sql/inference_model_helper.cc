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
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
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
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorData;
using ::tensorflow_federated::aggregation::TensorShape;

// Duplicates the data in a single column based on the per_row_output_counts.
// T is the C++ type of the data in the column.
template <typename T>
void DuplicateVectorData(const Tensor& original_column,
                         size_t original_row_count,
                         const std::vector<size_t>& per_row_output_counts,
                         TensorData* new_data_ptr) {
  auto* new_data = static_cast<MutableVectorData<T>*>(new_data_ptr);
  const auto original_span = original_column.AsSpan<T>();
  for (size_t i = 0; i < original_row_count; ++i) {
    const T& val = original_span[i];
    for (size_t k = 0; k < per_row_output_counts[i]; ++k) {
      new_data->push_back(val);
    }
  }
}

// Overload for DT_STRING.
void DuplicateStringData(const Tensor& original_column,
                         size_t original_row_count,
                         const std::vector<size_t>& per_row_output_counts,
                         TensorData* new_data_ptr) {
  auto* new_data = static_cast<MutableStringData*>(new_data_ptr);
  const auto original_span = original_column.AsSpan<absl::string_view>();
  for (size_t i = 0; i < original_row_count; ++i) {
    const absl::string_view val = original_span[i];
    for (size_t k = 0; k < per_row_output_counts[i]; ++k) {
      new_data->Add(std::string(val));
    }
  }
}

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

  switch (prompt.parser()) {
    case Prompt::PARSER_DELIMITER: {
      std::string delimiter = ",";
      if (!prompt.delimiter_parser().delimiter().empty()) {
        delimiter = prompt.delimiter_parser().delimiter();
      }
      std::vector<std::string> values =
          absl::StrSplit(inference_output, delimiter);
      for (std::string& value : values) {
        if (regex.has_value()) {
          output_string_data->Add(RegexMatch(value, *regex));
        } else {
          output_string_data->Add(std::move(value));
        }
      }
      return values.size();
    }
    case Prompt::PARSER_AUTO: {
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
      absl::Status status =
          JsonStringToMessage(json_string, &json_struct, options);

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
        std::string string_element;
        switch (element.kind_case()) {
          case Value::kStringValue:
            string_element = element.string_value();
            break;
          case Value::kNumberValue:
            string_element = absl::StrCat(element.number_value());
            break;
          case Value::kBoolValue:
            string_element = element.bool_value() ? "true" : "false";
            break;
          default:
            return absl::InvalidArgumentError(absl::StrCat(
                "JSON array contains an unsupported element of type ",
                element.kind_case()));
        }
        if (regex.has_value()) {
          output_string_data->Add(RegexMatch(string_element, *regex));
        } else {
          output_string_data->Add(std::move(string_element));
        }
        num_values_added++;
      }
      return num_values_added;
    }
    default: {  // PARSER_NONE or unspecified value
      std::string value = std::move(inference_output);
      if (regex.has_value()) {
        value = RegexMatch(value, *regex);
      }
      output_string_data->Add(std::move(value));
      return 1;
    }
  }
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

absl::StatusOr<std::vector<Tensor>> DuplicateTensorRows(
    const std::vector<Tensor>& original_columns, size_t original_row_count,
    const std::vector<size_t>& per_row_output_counts) {
  size_t total_new_rows = 0;
  for (size_t count : per_row_output_counts) {
    total_new_rows += count;
  }

  std::vector<std::unique_ptr<TensorData>> new_data_vec;
  new_data_vec.reserve(original_columns.size());
  for (const auto& col : original_columns) {
    switch (col.dtype()) {
      case DataType::DT_STRING: {
        auto new_data = std::make_unique<
            tensorflow_federated::aggregation::MutableStringData>(
            total_new_rows);
        DuplicateStringData(col, original_row_count, per_row_output_counts,
                            new_data.get());
        new_data_vec.push_back(std::move(new_data));
        break;
      }
      case DataType::DT_INT64: {
        auto new_data = std::make_unique<MutableVectorData<int64_t>>();
        new_data->reserve(total_new_rows);
        DuplicateVectorData<int64_t>(col, original_row_count,
                                     per_row_output_counts, new_data.get());
        new_data_vec.push_back(std::move(new_data));
        break;
      }
      case DataType::DT_INT32: {
        auto new_data = std::make_unique<MutableVectorData<int32_t>>();
        new_data->reserve(total_new_rows);
        DuplicateVectorData<int32_t>(col, original_row_count,
                                     per_row_output_counts, new_data.get());
        new_data_vec.push_back(std::move(new_data));
        break;
      }
      case DataType::DT_FLOAT: {
        auto new_data = std::make_unique<MutableVectorData<float>>();
        new_data->reserve(total_new_rows);
        DuplicateVectorData<float>(col, original_row_count,
                                   per_row_output_counts, new_data.get());
        new_data_vec.push_back(std::move(new_data));
        break;
      }
      case DataType::DT_DOUBLE: {
        auto new_data = std::make_unique<MutableVectorData<double>>();
        new_data->reserve(total_new_rows);
        DuplicateVectorData<double>(col, original_row_count,
                                    per_row_output_counts, new_data.get());
        new_data_vec.push_back(std::move(new_data));
        break;
      }
      default:
        return absl::UnimplementedError(
            absl::StrCat("Unsupported data type for duplication: ",
                         DataType_Name(col.dtype())));
    }
  }

  // Wrap each expanded column (new_data_vec[i]) data into a Tensor object.
  std::vector<Tensor> result;
  result.reserve(original_columns.size());
  for (size_t i = 0; i < original_columns.size(); ++i) {
    absl::StatusOr<Tensor> t =
        Tensor::Create(original_columns[i].dtype(),
                       TensorShape({static_cast<long>(total_new_rows)}),
                       std::move(new_data_vec[i]), original_columns[i].name());
    if (!t.ok()) {
      return t.status();
    }
    result.push_back(std::move(*t));
  }
  return result;
}

}  // namespace confidential_federated_compute::fed_sql
