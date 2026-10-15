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

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "containers/common/inference/inference_model_helper.h"
#include "containers/common/io/tabular/input.h"
#include "containers/common/local_inference/inference_model.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::fcp::confidentialcompute::RuntimeConfig;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

// The max prompt size is set to match the gemma.cpp default seq_len value.
constexpr size_t kMaxPromptSize = 8192;

// Duplicates rows in the input columns to match the number of rows in the
// output columns, which may be greater than one per input row. For now, only
// one output column duplication is supported (only supports a single output
// column in the per_row_output_counts_map). Once this is extended to multiple
// output columns, this will handle the cartesian product of multiple outputs.
absl::Status DuplicateColumnsForMultipleRows(
    Input& input, std::map<std::string, Tensor>& output_columns,
    const std::map<std::string, std::vector<size_t>>&
        per_row_output_counts_map) {
  if (output_columns.size() != 1) {
    // This function is called per-task, so there is always one output column.
    return absl::InternalError(
        "DuplicateColumnsForMultipleRows called with multiple output columns");
  }
  const size_t original_row_count = input.GetRowCount();
  if (original_row_count == 0) {
    return absl::OkStatus();
  }

  std::string metadata = std::string(input.GetMetadata());

  ABSL_ASSIGN_OR_RETURN(std::vector<Tensor> original_columns,
                        std::move(input).MoveToTensors());

  const auto& per_row_output_counts = per_row_output_counts_map.begin()->second;

  // Use the shared helper for tensor row duplication.
  ABSL_ASSIGN_OR_RETURN(
      std::vector<Tensor> final_columns,
      inference::DuplicateTensorRows(original_columns, original_row_count,
                                     per_row_output_counts));

  ABSL_ASSIGN_OR_RETURN(
      Input new_input,
      Input::CreateFromTensors(std::move(final_columns), std::move(metadata)));
  input = std::move(new_input);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<InferenceRunner> InferenceRunner::BuildModel(
    const local_inference::SessionInferenceConfiguration&
        inference_configuration) {
  ABSL_ASSIGN_OR_RETURN(
      local_inference::InferenceModel model,
      local_inference::InferenceModel::BuildModel(inference_configuration));
  return InferenceRunner(
      std::make_unique<local_inference::InferenceModel>(std::move(model)));
}

const local_inference::SessionInferenceConfiguration&
InferenceRunner::GetInferenceConfiguration() const {
  return model_->GetInferenceConfiguration();
}

absl::Status InferenceRunner::RunInference(Input& input) {
  const local_inference::SessionInferenceConfiguration&
      inference_configuration = model_->GetInferenceConfiguration();
  const RuntimeConfig& runtime_config =
      inference_configuration.initialize_configuration.inference_config()
          .runtime_config();
  const size_t max_prompt_size = runtime_config.max_prompt_size() > 0
                                     ? runtime_config.max_prompt_size()
                                     : kMaxPromptSize;

  for (const auto& inference_task :
       inference_configuration.initialize_configuration.inference_config()
           .inference_task()) {
    if (!inference_task.has_prompt()) {
      return absl::InvalidArgumentError(
          "Prompt not found when running inference. Only prompt-based "
          "inference is supported.");
    }
    std::vector<std::string> input_column_names;
    if (!inference_task.column_config().input_column_names().empty()) {
      input_column_names.insert(
          input_column_names.end(),
          inference_task.column_config().input_column_names().begin(),
          inference_task.column_config().input_column_names().end());
    } else {
      return absl::InvalidArgumentError(
          "No input column names found when running inference.");
    }
    const std::string& output_column_name =
        inference_task.column_config().output_column_name();

    // Find the indices that correspond to the input columns.
    absl::Span<const std::string> column_names = input.GetColumnNames();
    std::vector<size_t> input_column_indices;
    for (const auto& input_column_name : input_column_names) {
      const auto it = std::find(column_names.begin(), column_names.end(),
                                input_column_name);
      if (it == column_names.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Couldn't find an input column ", input_column_name,
                         " to run inference on."));
      }
      input_column_indices.push_back(it - column_names.begin());
    }

    // For each row of the input columns, populate the prompt template to
    // create a combined prompt matching the column values.
    const size_t row_count = input.GetRowCount();
    std::vector<std::string> prompts;
    prompts.reserve(row_count);
    for (size_t i = 0; i < row_count; ++i) {
      ABSL_ASSIGN_OR_RETURN(RowView row, input.GetRow(i));
      ABSL_ASSIGN_OR_RETURN(
          std::string combined_prompt,
          prompt_processor_.PopulatePromptTemplate(
              inference_task.prompt(), row, input.GetColumnNames(),
              input_column_indices, output_column_name, max_prompt_size));
      prompts.push_back(std::move(combined_prompt));
    }

    // GenerateBatch chunks internally by RuntimeConfig.max_batch_size.
    std::vector<absl::StatusOr<std::string>> generated =
        model_->GenerateBatch(prompts);

    std::unique_ptr<MutableStringData> output_string_data =
        std::make_unique<MutableStringData>(static_cast<long>(row_count));
    size_t num_output_rows = 0;
    std::vector<size_t> per_row_output_counts;
    per_row_output_counts.reserve(row_count);
    for (absl::StatusOr<std::string>& output_string : generated) {
      // A generation failure aborts the whole session. Only output
      // *processing* failures are tolerated and filled in with empty strings.
      ABSL_RETURN_IF_ERROR(output_string.status());
      absl::StatusOr<size_t> num_rows_added_status =
          output_processor_.ProcessInferenceOutput(
              inference_task.prompt(), std::move(*output_string),
              output_column_name, output_string_data.get());
      if (!num_rows_added_status.ok()) {
        LOG(WARNING) << "Failed to process inference output: "
                     << num_rows_added_status.status()
                     << ". Outputting empty string.";
        output_string_data->Add("");
        num_output_rows++;
        per_row_output_counts.push_back(1);
      } else {
        num_output_rows += *num_rows_added_status;
        per_row_output_counts.push_back(*num_rows_added_status);
      }
    }

    ABSL_ASSIGN_OR_RETURN(
        Tensor output_tensor,
        Tensor::Create(DataType::DT_STRING,
                       TensorShape({static_cast<long>(num_output_rows)}),
                       std::move(output_string_data), output_column_name));

    std::map<std::string, Tensor> current_output_column;
    current_output_column.emplace(output_column_name, std::move(output_tensor));

    // If any of the rows generated multiple outputs, duplicate the columns to
    // match the new row count.
    if (std::any_of(per_row_output_counts.begin(), per_row_output_counts.end(),
                    [](size_t count) { return count != 1; })) {
      std::map<std::string, std::vector<size_t>> per_row_output_counts_map;
      per_row_output_counts_map.emplace(output_column_name,
                                        std::move(per_row_output_counts));

      ABSL_RETURN_IF_ERROR(DuplicateColumnsForMultipleRows(
          input, current_output_column, per_row_output_counts_map));
    }

    ABSL_RETURN_IF_ERROR(
        input.AddColumn(std::move(current_output_column.begin()->second)));
  }

  return absl::OkStatus();
}

}  // namespace confidential_federated_compute::fed_sql
