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

#include "containers/batched_inference/batched_inference_fn.h"

#include <algorithm>
#include <queue>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "containers/batched_inference/batched_inference_engine.h"
#include "containers/common/input.h"
#include "containers/common/row_set.h"
#include "containers/fed_sql/inference_model_helper.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/fns/do_fn.h"
#include "containers/fns/fn.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "google/protobuf/any.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::batched_inference {
namespace {

// TODO:
//
// - Classify potential types of errors based on criticality and revise error
// types here to
//   match that classification. Thread through the DoFn framework, such that
//   non-critical errors do not abort the session, and can be handled in Flume
//   on the host to update /streamz counters, etc.
//
// - Port the FedSql logic here, in particular the support for inference
// returning zero or more
//   than one value for an output column in a row (to be handled as repeats,
//   outer join, etc.).
//
// - Switch the semantics to allow early processing before the Commit(), and
// during a Write() if
//   there's enough to fill an inference batch.
//
// - Add support for handling protobuf inputs (that require a message factory),
// to match FedSql.
//
// - ...

using ::confidential_federated_compute::fed_sql::Deserialize;
using ::confidential_federated_compute::fed_sql::InferenceOutputProcessor;
using ::confidential_federated_compute::fed_sql::InferencePromptProcessor;
using ::fcp::confidentialcompute::InferenceConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::InferenceTask;
using ::fcp::confidentialcompute::TableSchema;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorData;
using ::tensorflow_federated::aggregation::TensorShape;

// Represents a work item at the level of a single inference call.
struct CallLevelWorkItem {
  explicit CallLevelWorkItem(std::string prompt) : prompt(std::move(prompt)) {}

  std::string prompt;
  std::string result;
};

// Represents a work item at the level of a single inference task.
struct TaskLevelWorkItem {
  explicit TaskLevelWorkItem(InferenceTask task) : task(std::move(task)) {}

  InferenceTask task;
  std::vector<std::string> input_column_names;
  std::string output_column_name;
  std::vector<std::unique_ptr<CallLevelWorkItem>> call_items;
};

// Represents a work item at the level of a single blob (but maybe
// multiple inference tasks if multiple are defined).
struct BlobLevelWorkItem {
  explicit BlobLevelWorkItem(google::protobuf::Any key, std::string blob_id)
      : key(std::move(key)), blob_id(std::move(blob_id)) {}

  google::protobuf::Any key;
  std::string blob_id;
  std::unique_ptr<Input> input;
  std::vector<std::unique_ptr<TaskLevelWorkItem>> task_items;
};

// The parsed string values for one output column of an inference task.
struct ParsedColumn {
  std::string name;
  // values[i] is the list of strings extracted from the LLM output for
  // input row i.
  std::vector<std::vector<std::string>> values;
};

// Holds the processed output for a single inference task.
struct TaskOutput {
  // Parsed string values per output column, populated by ProcessTaskOutputs.
  // Currently each task has 1 ParsedColumn. but using a vector allows future
  // extension to multi-column output.
  std::vector<ParsedColumn> parsed_columns;
  // Finalized output tensors, populated by CartesianExpand. Currently each task
  // produces a single output column, but using a vector allows future extension
  // to multi-column output.
  std::vector<Tensor> output_columns;
  // The number of output values generated for each input row. For example,
  // {1, 2, 1} means row 0 produced 1 output, row 1 produced 2, row 2 produced
  // 1. For PARSER_NONE this is always all 1s.
  std::vector<size_t> per_row_output_counts;
  // True if any entry in per_row_output_counts is not 1.
  bool HasMultiRowOutput() const {
    return std::any_of(per_row_output_counts.begin(),
                       per_row_output_counts.end(),
                       [](size_t count) { return count != 1; });
  }
};

// Unpacks a single inference tqask into a set of input columns and an output.
absl::Status UnpackTasksForBlob(const InferenceConfiguration& inference_config,
                                BlobLevelWorkItem* blob_item) {
  for (const auto& inference_task : inference_config.inference_task()) {
    if (!inference_task.has_prompt()) {
      return absl::InvalidArgumentError(
          "Prompt not found when running inference. Only prompt-based "
          "inference is supported.");
    }
    if (inference_task.column_config().input_column_names().empty()) {
      return absl::InvalidArgumentError(
          "No input column names found when running inference.");
    }
    auto task_item = std::make_unique<TaskLevelWorkItem>(inference_task);
    task_item->input_column_names.insert(
        task_item->input_column_names.end(),
        inference_task.column_config().input_column_names().begin(),
        inference_task.column_config().input_column_names().end());
    task_item->output_column_name =
        inference_task.column_config().output_column_name();
    blob_item->task_items.push_back(std::move(task_item));
  }
  return absl::OkStatus();
}

TableSchema CreateSchemaFromTasks(const BlobLevelWorkItem& blob_item) {
  absl::flat_hash_set<std::string> columns;
  for (const auto& task_item : blob_item.task_items) {
    if (!task_item->output_column_name.empty()) {
      columns.insert(task_item->output_column_name);
    }
  }
  TableSchema schema;
  for (const std::string& col_name : columns) {
    schema.add_column()->set_name(col_name);
  }
  return schema;
}

absl::Status UnpackCallsForTask(const Input& input, size_t max_prompt_size,
                                TaskLevelWorkItem* task_item) {
  std::vector<size_t> input_column_indices;
  for (const auto& input_column_name : task_item->input_column_names) {
    const auto it = std::find(input.GetColumnNames().begin(),
                              input.GetColumnNames().end(), input_column_name);
    if (it == input.GetColumnNames().end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Couldn't find an input column ", input_column_name,
                       " to run inference on."));
    }
    input_column_indices.push_back(it - input.GetColumnNames().begin());
  }
  InferencePromptProcessor prompt_processor;
  for (int i = 0; i < input.GetRowCount(); ++i) {
    absl::StatusOr<RowView> row_or = input.GetRow(i);
    if (!row_or.ok()) {
      return absl::InternalError(
          absl::StrCat("Couldn't get a row: ", row_or.status()));
    }
    absl::StatusOr<std::string> prompt_or =
        prompt_processor.PopulatePromptTemplate(
            task_item->task.prompt(), *row_or, input.GetColumnNames(),
            input_column_indices, task_item->output_column_name,
            max_prompt_size);
    if (!prompt_or.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Couldn't get a prompt: ", prompt_or.status()));
    }
    task_item->call_items.push_back(
        std::make_unique<CallLevelWorkItem>(std::move(*prompt_or)));
  }
  return absl::OkStatus();
}

// Processes the raw inference results for all tasks in a blob, parsing each
// result using the configured parser and tracking how many output values each
// input row produced. Returns a vector of TaskOutput, one per inference task.
// Only the parsed_columns and per_row_output_counts in each TaskOutput are
// populated. output_columns is left empty, to be populated later by
// CartesianExpand.
absl::StatusOr<std::vector<TaskOutput>> ProcessTaskOutputs(
    BlobLevelWorkItem* blob_item) {
  const long num_rows = static_cast<long>(blob_item->input->GetRowCount());
  std::vector<TaskOutput> task_outputs;
  InferenceOutputProcessor output_processor;

  for (auto& task_item : blob_item->task_items) {
    TaskOutput task_output;
    ParsedColumn parsed_column;
    parsed_column.name = task_item->output_column_name;
    parsed_column.values.reserve(num_rows);
    task_output.per_row_output_counts.reserve(num_rows);

    // Process each row of the inference result through the configured parser.
    for (auto& call_item : task_item->call_items) {
      // Use a fresh MutableStringData per input row so we can isolate this
      // row's parsed values.
      auto row_data = std::make_unique<MutableStringData>(0);
      absl::StatusOr<size_t> process_result =
          output_processor.ProcessInferenceOutput(
              task_item->task.prompt(), std::move(call_item->result),
              task_item->output_column_name, row_data.get());
      if (!process_result.ok()) {
        return absl::InternalError(absl::StrCat(
            "Couldn't process inference output: ", process_result.status()));
      }

      // Extract strings from MutableStringData via a temporary Tensor.
      size_t count = *process_result;
      absl::StatusOr<Tensor> temp_tensor = Tensor::Create(
          DataType::DT_STRING, TensorShape({static_cast<long>(count)}),
          std::move(row_data), task_item->output_column_name);
      if (!temp_tensor.ok()) {
        return absl::InternalError(absl::StrCat(
            "Couldn't create temporary tensor for column ",
            task_item->output_column_name, ": ", temp_tensor.status()));
      }
      auto span = temp_tensor->AsSpan<absl::string_view>();
      std::vector<std::string> row_values(span.begin(), span.end());

      parsed_column.values.push_back(std::move(row_values));
      task_output.per_row_output_counts.push_back(count);
    }

    task_output.parsed_columns.push_back(std::move(parsed_column));
    task_outputs.push_back(std::move(task_output));
  }

  return task_outputs;
}

// Computes the Cartesian product of parsed output values across all tasks for
// each input row. Populates output_columns on each TaskOutput and returns the
// final number of output rows generated per input row.
//
// If a task produces 0 values for a given input row, it is excluded from the
// Cartesian product for that row and its column is filled with empty strings.
// If all tasks produce 0 values for a row, all the inference output columns for
// that row are empty strings.
absl::StatusOr<std::vector<size_t>> CartesianExpand(
    std::vector<TaskOutput>& task_outputs, size_t num_input_rows) {
  const size_t num_tasks = task_outputs.size();

  // Create one MutableStringData collector per ParsedColumn across all tasks.
  // collectors[t] corresponds to task_outputs[t].parsed_columns[0].
  std::vector<std::unique_ptr<MutableStringData>> collectors;
  collectors.reserve(num_tasks);
  for (size_t t = 0; t < num_tasks; ++t) {
    collectors.push_back(std::make_unique<MutableStringData>(0));
  }

  std::vector<size_t> input_row_duplication_counts;
  input_row_duplication_counts.reserve(num_input_rows);

  for (size_t i = 0; i < num_input_rows; ++i) {
    // Identify which tasks produced values for this row (active tasks).
    std::vector<size_t> active_task_indices;
    for (size_t t = 0; t < num_tasks; ++t) {
      if (task_outputs[t].per_row_output_counts[i] > 0) {
        active_task_indices.push_back(t);
      }
    }

    // Generate the Cartesian product across `num_active` active tasks for
    // input row `i`,
    const size_t num_active = active_task_indices.size();
    // For input row i and active task a, sizes[a] is the number of output rows
    // task a produces for row i.
    std::vector<size_t> sizes(num_active);
    for (size_t a = 0; a < num_active; ++a) {
      sizes[a] = task_outputs[active_task_indices[a]].per_row_output_counts[i];
    }

    // For example, given 3 tasks with sizes = [1, 3, 2] for input row i, the
    // following while loop produces exactly 1 * 3 * 2 = 6 combinations:
    //   indices = [0, 0, 0]: 0-th output values for all 3 tasks.
    //   indices = [0, 0, 1]: 0-th output values for tasks 0 and 2, 1-st output
    //     value for task 1.
    //   indices = [0, 1, 0]
    //   indices = [0, 1, 1]
    //   indices = [0, 2, 0]
    //   indices = [0, 2, 1]

    // indices keep track of the current combination to extract values for.
    std::vector<size_t> indices(num_active, 0);
    size_t combinations = 0;
    while (true) {
      // Append the value corresponding to current indices into each task's
      // collector.
      for (size_t t = 0; t < num_tasks; ++t) {
        // Find if this task participates for this input row.
        bool is_active = false;
        size_t active_pos = 0;
        for (size_t a = 0; a < num_active; ++a) {
          if (active_task_indices[a] == t) {
            is_active = true;
            active_pos = a;
            break;
          }
        }
        if (is_active) {
          const std::string& val =
              task_outputs[t].parsed_columns[0].values[i][indices[active_pos]];
          collectors[t]->Add(std::string(val));
        } else {
          // Task produced 0 values for this row; fill with empty string.
          collectors[t]->Add(std::string(""));
        }
      }
      combinations++;

      // Increment the rightmost index. If it overflows (reaches its
      // size limit stored in `sizes`), reset it to 0 and carry over by moving
      // leftwards to the next index. Repeat until an index doesn't overflow,
      // or carry drops below 0 (meaning all combinations have been produced).
      int carry = static_cast<int>(num_active) - 1;
      while (carry >= 0) {
        indices[carry]++;
        if (indices[carry] < sizes[carry]) break;
        indices[carry] = 0;
        carry--;
      }
      if (carry < 0) break;  // All combinations exhausted.
    }

    input_row_duplication_counts.push_back(combinations);
  }

  // Precompute total output rows once, as it is identical across all tasks.
  size_t total_rows = 0;
  for (size_t count : input_row_duplication_counts) {
    total_rows += count;
  }

  // Wrap each collector into a Tensor and store in the corresponding
  // TaskOutput's output_columns.
  for (size_t t = 0; t < num_tasks; ++t) {
    absl::StatusOr<Tensor> tensor = Tensor::Create(
        DataType::DT_STRING, TensorShape({static_cast<long>(total_rows)}),
        std::move(collectors[t]), task_outputs[t].parsed_columns[0].name);
    if (!tensor.ok()) {
      return absl::InternalError(absl::StrCat(
          "Couldn't create output tensor for column ",
          task_outputs[t].parsed_columns[0].name, ": ", tensor.status()));
    }
    task_outputs[t].output_columns.push_back(std::move(*tensor));
  }

  return input_row_duplication_counts;
}

// A class that batches all writes received until a Commit(), and then
// issues a single call to the inference provider.
//
// Currently implemented behavior and restrictions:
// 1. One row in, one row out - we enforce the exact 1:1 correspondence.
// 2. If anything in the committed set of writes fails, the entire commit
//    fails as well; there are no silent errors (empty results).
//
// FUTURE WORK: Relax these restrictiosn. and port over the logic used in
// the FedSql container. Possibly make this behavior configurable.
class BatchedInferenceFn final
    : public confidential_federated_compute::fns::DoFn {
 public:
  explicit BatchedInferenceFn(
      std::shared_ptr<BatchedInferenceEngine> batched_inference_engine,
      InferenceConfiguration inference_config)
      : batched_inference_engine_(batched_inference_engine),
        inference_config_(std::move(inference_config)) {}

  ~BatchedInferenceFn() {}

  absl::Status Do(Session::KV kv, Context& context) override;

  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest commit_request,
      Context& context) override;

 private:
  absl::Status DoBatchedInferenceInternal(
      const std::vector<CallLevelWorkItem*>& batch);

  absl::Status FinalizeBlob(BlobLevelWorkItem* blob_item, Context& context);

  absl::Status FinalizeAllBlobs(std::queue<BlobLevelWorkItem*>* blob_items,
                                Context& context);

  absl::Status DoBatchedInferenceAndFinalizeAllBlobs(
      const std::vector<CallLevelWorkItem*>& batched_call_items,
      std::queue<BlobLevelWorkItem*>* blob_items, Context& context);

  std::shared_ptr<BatchedInferenceEngine> batched_inference_engine_;
  InferenceConfiguration inference_config_;
  std::vector<std::unique_ptr<BlobLevelWorkItem>> uncommitted_blob_items_;
};

absl::Status BatchedInferenceFn::Do(Session::KV kv, Context& context) {
  auto blob_item = std::make_unique<BlobLevelWorkItem>(kv.key, kv.blob_id);
  absl::Status task_unpack_status =
      UnpackTasksForBlob(inference_config_, blob_item.get());
  if (!task_unpack_status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to unpack inference tasks: ", task_unpack_status));
  }
  FederatedComputeCheckpointParserFactory parser_factory;
  absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
      parser_factory.Create(absl::Cord(std::move(kv.data)));
  if (!parser.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to construct a checkpoint parser: ", parser.status()));
  }
  TableSchema table_schema = CreateSchemaFromTasks(*blob_item);
  absl::StatusOr<std::vector<Tensor>> tensors_or =
      Deserialize(table_schema, parser->get(), inference_config_);
  if (!tensors_or.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to deserialize tensors from checkpoint: ",
                     tensors_or.status()));
  }
  auto input_or = Input::CreateFromTensors(std::move(*tensors_or));
  if (!input_or.ok()) {
    return absl::InternalError(absl::StrCat(
        "Failed to create input from tensors: ", input_or.status()));
  }
  blob_item->input = std::make_unique<Input>(std::move(*input_or));
  for (auto& task_item : blob_item->task_items) {
    absl::Status call_unpack_status = UnpackCallsForTask(
        *blob_item->input, inference_config_.runtime_config().max_prompt_size(),
        task_item.get());
    if (!call_unpack_status.ok()) {
      return absl::InternalError(absl::StrCat(
          "Failed to unpack calls for inference task: ", call_unpack_status));
    }
  }
  uncommitted_blob_items_.push_back(std::move(blob_item));
  return absl::OkStatus();
}

absl::StatusOr<fcp::confidentialcompute::CommitResponse>
BatchedInferenceFn::Commit(
    fcp::confidentialcompute::CommitRequest commit_request, Context& context) {
  std::vector<std::unique_ptr<BlobLevelWorkItem>> uncommitted_blob_items;
  std::swap(uncommitted_blob_items, uncommitted_blob_items_);
  const int batch_size =
      std::max(1, inference_config_.runtime_config().max_batch_size());
  std::queue<BlobLevelWorkItem*> pending_blob_items;
  std::vector<CallLevelWorkItem*> batched_call_items;
  for (auto& blob_item : uncommitted_blob_items) {
    for (auto& task_item : blob_item->task_items) {
      for (auto& call_item : task_item->call_items) {
        batched_call_items.push_back(call_item.get());
        if (batched_call_items.size() > batch_size) {
          return absl::InternalError(absl::StrCat(
              "Batching logic failed, resulting in ", batched_call_items.size(),
              " iterms for batch size of ", batch_size));
        }
        if (batched_call_items.size() == batch_size) {
          absl::Status inference_and_finalize_status =
              DoBatchedInferenceAndFinalizeAllBlobs(
                  batched_call_items, &pending_blob_items, context);
          if (!inference_and_finalize_status.ok()) {
            return inference_and_finalize_status;
          }
          batched_call_items.clear();
        }
      }
    }
    if (batched_call_items.empty()) {
      absl::Status finalize_status = FinalizeBlob(blob_item.get(), context);
      if (!finalize_status.ok()) {
        return finalize_status;
      }
    } else {
      pending_blob_items.push(blob_item.get());
    }
  }
  if (!batched_call_items.empty()) {
    absl::Status inference_and_finalize_status =
        DoBatchedInferenceAndFinalizeAllBlobs(batched_call_items,
                                              &pending_blob_items, context);
    if (!inference_and_finalize_status.ok()) {
      return inference_and_finalize_status;
    }
  }
  return fcp::confidentialcompute::CommitResponse();
}

absl::Status BatchedInferenceFn::DoBatchedInferenceAndFinalizeAllBlobs(
    const std::vector<CallLevelWorkItem*>& batched_call_items,
    std::queue<BlobLevelWorkItem*>* blob_items, Context& context) {
  absl::Status inference_status =
      DoBatchedInferenceInternal(batched_call_items);
  if (!inference_status.ok()) {
    return inference_status;
  }
  absl::Status finalize_status = FinalizeAllBlobs(blob_items, context);
  if (!finalize_status.ok()) {
    return finalize_status;
  }
  return absl::OkStatus();
}

absl::Status BatchedInferenceFn::DoBatchedInferenceInternal(
    const std::vector<CallLevelWorkItem*>& batch) {
  std::vector<std::string> prompts;
  for (CallLevelWorkItem* call_item : batch) {
    prompts.push_back(call_item->prompt);
  }
  std::vector<absl::StatusOr<std::string>> results =
      batched_inference_engine_->DoBatchedInference(prompts);
  if (results.size() != prompts.size()) {
    return absl::InternalError(absl::StrCat(
        "The number of results (", results.size(),
        ") does not match the number of prompts (", prompts.size(), ")."));
  }
  for (int i = 0; i < results.size(); ++i) {
    if (!results[i].ok()) {
      return absl::InternalError(
          absl::StrCat("Inference failed: ", results[i].status()));
    }
    batch[i]->result = *results[i];
  }
  return absl::OkStatus();
}

absl::Status BatchedInferenceFn::FinalizeBlob(BlobLevelWorkItem* blob_item,
                                              Context& context) {
  const long num_rows = static_cast<long>(blob_item->input->GetRowCount());

  // Process all task outputs and parse each inference result.
  absl::StatusOr<std::vector<TaskOutput>> task_outputs_or =
      ProcessTaskOutputs(blob_item);
  if (!task_outputs_or.ok()) {
    return task_outputs_or.status();
  }
  std::vector<TaskOutput>& task_outputs = *task_outputs_or;

  // Compute the Cartesian product across tasks and populate output column
  // tensors.
  FCP_ASSIGN_OR_RETURN(std::vector<size_t> input_row_duplication_counts,
                       CartesianExpand(task_outputs, num_rows));

  // Extract input tensors from the input blob.
  absl::StatusOr<std::vector<Tensor>> input_tensors =
      std::move(*blob_item->input).MoveToTensors();
  if (!input_tensors.ok()) {
    return absl::InternalError(absl::StrCat("Couldn't recover input tensors: ",
                                            input_tensors.status()));
  }

  // Duplicate input tensors according to our active Cartesian expansion
  // profile.
  absl::StatusOr<std::vector<Tensor>> duplicated = fed_sql::DuplicateTensorRows(
      *input_tensors, num_rows, input_row_duplication_counts);
  if (!duplicated.ok()) {
    return absl::InternalError(absl::StrCat(
        "Couldn't duplicate input tensors: ", duplicated.status()));
  }
  *input_tensors = std::move(*duplicated);

  // Package possibly duplicated input tensors into a checkpoint.
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  for (auto& tensor : *input_tensors) {
    absl::Status add_status =
        checkpoint_builder->Add(tensor.name(), std::move(tensor));
    if (!add_status.ok()) {
      return absl::InternalError(
          absl::StrCat("Couldn't add input tensor: ", tensor.name()));
    }
  }

  // Add output tensors to the checkpoint.
  for (auto& task_output : task_outputs) {
    for (auto& tensor : task_output.output_columns) {
      absl::Status add_status =
          checkpoint_builder->Add(tensor.name(), std::move(tensor));
      if (!add_status.ok()) {
        return absl::InternalError(
            absl::StrCat("Couldn't add output tensor: ", tensor.name()));
      }
    }
  }

  // Build the final checkpoint and emit it as encrypted output.
  absl::StatusOr<absl::Cord> checkpoint_or = checkpoint_builder->Build();
  if (!checkpoint_or.ok()) {
    return absl::InternalError(
        absl::StrCat("Couldn't build checkpoint: ", checkpoint_or.status()));
  }
  // NOTE: We are intentionally using the same blob ids in the output as
  // those used in the corresponding inputs.
  // FUTURE WORK(b/452094015): Consider making this behavior configurable
  // if needed.
  std::string flattened_output(checkpoint_or->Flatten());
  if (!context.EmitEncrypted(
          0, Session::KV{blob_item->key, std::move(flattened_output),
                         blob_item->blob_id})) {
    return absl::InternalError("EmitEncrypted failed");
  }
  return absl::OkStatus();
}

absl::Status BatchedInferenceFn::FinalizeAllBlobs(
    std::queue<BlobLevelWorkItem*>* blob_items, Context& context) {
  while (!blob_items->empty()) {
    absl::Status status = FinalizeBlob(blob_items->front(), context);
    if (!status.ok()) {
      return status;
    }
    blob_items->pop();
  }
  return absl::OkStatus();
}

class BatchedInferenceFnFactory
    : public confidential_federated_compute::fns::FnFactory {
 public:
  explicit BatchedInferenceFnFactory(
      std::shared_ptr<BatchedInferenceEngine> batched_inference_engine,
      InferenceConfiguration inference_config)
      : batched_inference_engine_(batched_inference_engine),
        inference_config_(std::move(inference_config)) {}

  absl::StatusOr<std::unique_ptr<fns::Fn>> CreateFn() const override {
    return std::make_unique<BatchedInferenceFn>(batched_inference_engine_,
                                                inference_config_);
  }

 private:
  std::shared_ptr<BatchedInferenceEngine> batched_inference_engine_;
  InferenceConfiguration inference_config_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<fns::FnFactory>> CreateBatchedInferenceFnFactory(
    std::shared_ptr<BatchedInferenceEngine> batched_inference_engine,
    InferenceConfiguration inference_config) {
  return std::make_unique<BatchedInferenceFnFactory>(
      batched_inference_engine, std::move(inference_config));
}

}  // namespace confidential_federated_compute::batched_inference