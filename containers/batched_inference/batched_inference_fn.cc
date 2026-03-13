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
#include "containers/batched_inference/batched_inference_provider.h"
#include "containers/fed_sql/inference_model_helper.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/fns/do_fn.h"
#include "containers/fns/fn.h"
#include "containers/sql/input.h"
#include "containers/sql/row_set.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "google/protobuf/any.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::batched_inference {
namespace {

using ::confidential_federated_compute::Session;
using ::confidential_federated_compute::fed_sql::
    CreateInputFromMessageCheckpoint;
using ::confidential_federated_compute::fed_sql::Deserialize;
using ::confidential_federated_compute::fed_sql::InferenceOutputProcessor;
using ::confidential_federated_compute::fed_sql::InferencePromptProcessor;
using ::confidential_federated_compute::sql::Input;
using ::confidential_federated_compute::sql::RowLocation;
using ::confidential_federated_compute::sql::RowSet;
using ::confidential_federated_compute::sql::RowView;
using ::fcp::confidentialcompute::BlobHeader;
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
      std::shared_ptr<BatchedInferenceProvider> batched_inference_provider,
      InferenceConfiguration inference_config)
      : batched_inference_provider_(batched_inference_provider),
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

  std::shared_ptr<BatchedInferenceProvider> batched_inference_provider_;
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
  auto input_or =
      Input::CreateFromTensors(std::move(*tensors_or), BlobHeader());
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
      batched_inference_provider_->DoBatchedInference(prompts);
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
  absl::StatusOr<std::vector<Tensor>> tensors_or =
      std::move(*blob_item->input).MoveToTensors();
  if (!tensors_or.ok()) {
    return absl::InternalError(
        absl::StrCat("Couldn't recover input tensors: ", tensors_or.status()));
  }
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  for (auto& tensor : *tensors_or) {
    absl::Status add_status =
        checkpoint_builder->Add(tensor.name(), std::move(tensor));
    if (!add_status.ok()) {
      return absl::InternalError(
          absl::StrCat("Couldn't add input tensor: ", tensor.name()));
    }
  }
  InferenceOutputProcessor output_processor;
  for (auto& task_item : blob_item->task_items) {
    auto output_string_data = std::make_unique<MutableStringData>(num_rows);
    for (auto& call_item : task_item->call_items) {
      absl::StatusOr<size_t> process_result =
          output_processor.ProcessInferenceOutput(
              task_item->task.prompt(), std::move(call_item->result),
              task_item->output_column_name, output_string_data.get());
      if (!process_result.ok()) {
        return absl::InternalError(absl::StrCat(
            "Couldn't process inference output: ", process_result.status()));
      }
      if (*process_result != 1) {
        return absl::InternalError(
            absl::StrCat("Processing of inference output generated more or "
                         "less than one row: ",
                         *process_result));
      }
    }
    absl::StatusOr<Tensor> output_tensor_or = Tensor::Create(
        DataType::DT_STRING, TensorShape({num_rows}),
        std::move(output_string_data), task_item->output_column_name);
    if (!output_tensor_or.ok()) {
      return absl::InternalError(absl::StrCat(
          "Couldn't create an output tensor for column ",
          task_item->output_column_name, ": ", output_tensor_or.status()));
    }
    absl::Status add_status = checkpoint_builder->Add(
        task_item->output_column_name, std::move(*output_tensor_or));
    if (!add_status.ok()) {
      return absl::InternalError(absl::StrCat("Couldn't add output tensor: ",
                                              task_item->output_column_name));
    }
  }
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
      std::shared_ptr<BatchedInferenceProvider> batched_inference_provider,
      InferenceConfiguration inference_config)
      : batched_inference_provider_(batched_inference_provider),
        inference_config_(std::move(inference_config)) {}

  absl::StatusOr<std::unique_ptr<fns::Fn>> CreateFn() const override {
    return std::make_unique<BatchedInferenceFn>(batched_inference_provider_,
                                                inference_config_);
  }

 private:
  std::shared_ptr<BatchedInferenceProvider> batched_inference_provider_;
  InferenceConfiguration inference_config_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<fns::FnFactory>> CreateBatchedInferenceFnFactory(
    std::shared_ptr<BatchedInferenceProvider> batched_inference_provider,
    InferenceConfiguration inference_config) {
  return std::make_unique<BatchedInferenceFnFactory>(
      batched_inference_provider, std::move(inference_config));
}

}  // namespace confidential_federated_compute::batched_inference