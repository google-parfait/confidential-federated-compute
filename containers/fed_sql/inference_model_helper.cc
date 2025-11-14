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

#include "absl/status/status.h"
#include "containers/sql/input.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::confidential_federated_compute::sql::RowView;
using ::fcp::confidentialcompute::Prompt;
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

absl::StatusOr<std::string> InferenceOutputProcessor::ApplyRegex(
    const Prompt& prompt, const std::string& output_string) {
  if (!prompt.regex().empty()) {
    try {
      std::regex regex(prompt.regex());
      return RegexMatch(output_string, regex);
    } catch (const std::regex_error& e) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid regex provided in prompt: '", prompt.regex(),
                       "': ", e.what()));
    }
  }
  return output_string;
}

InferencePromptProcessor::InferencePromptProcessor() {}

absl::StatusOr<std::string> InferencePromptProcessor::PopulatePromptTemplate(
    const std::string& prompt_template, const RowView& row,
    absl::Span<const std::string> column_names,
    absl::Span<const size_t> input_column_indices) {
  std::string populated_prompt(prompt_template);
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
  return populated_prompt;
}

}  // namespace confidential_federated_compute::fed_sql
