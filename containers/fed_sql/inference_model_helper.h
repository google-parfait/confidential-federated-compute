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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_MODEL_HELPER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_MODEL_HELPER_H_

#include <regex>
#include <string>

#include "absl/status/status.h"
#include "containers/sql/input.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"

namespace confidential_federated_compute::fed_sql {

// A helper class for processing the outputs of one inference task.
class InferenceOutputProcessor {
 public:
  InferenceOutputProcessor();

  // Applies the regex from the Prompt configuration to the output string.
  // Returns the modified string if a match is found, otherwise returns the
  // original string.
  absl::StatusOr<std::string> ApplyRegex(
      const fcp::confidentialcompute::Prompt& prompt,
      const std::string& output_string);

 private:
  // Apply a regex matching to the given text. Returns only the first match. If
  // no match is found, returns the original text.
  std::string RegexMatch(const std::string& text, const std::regex& regex);
};

class InferencePromptProcessor {
 public:
  InferencePromptProcessor();
  // Build the combined prompt for a single row of data by populating the
  // template and replacing the placeholders with the column values.
  absl::StatusOr<std::string> PopulatePromptTemplate(
      const fcp::confidentialcompute::Prompt& prompt, const sql::RowView& row,
      absl::Span<const std::string> column_names,
      absl::Span<const size_t> input_column_indices,
      const std::string& output_column_name, size_t max_prompt_size);

 private:
  // A helper function to append system instructions to the prompt if the prompt
  // config is set to PARSER_AUTO.
  void AppendSystemInstructions(std::string& prompt,
                                const std::string& output_column_name);
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_MODEL_HELPER_H_
