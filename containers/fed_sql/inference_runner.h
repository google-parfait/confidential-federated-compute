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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_RUNNER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_RUNNER_H_

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/common/inference/inference_model_helper.h"
#include "containers/common/io/tabular/input.h"
#include "containers/common/local_inference/inference_model.h"

namespace confidential_federated_compute::fed_sql {

// Runs prompt-based inference over tabular data.
//
// This is the tabular wrapper around local_inference::InferenceModel, which is
// a pure generation engine. InferenceRunner owns everything the engine
// deliberately does not: rendering prompt templates from column values,
// parsing model output, and expanding rows when a single input row produces
// multiple output values.
//
// This wrapper is fed_sql-specific on purpose. The batched-inference
// containers drive the same engine through BatchedInferenceFn instead, which
// implements its own (subtly different) templating and expansion semantics.
class InferenceRunner {
 public:
  // Constructs an InferenceRunner wrapping `model` (typically a
  // MockInferenceModel in tests).
  explicit InferenceRunner(
      std::unique_ptr<local_inference::InferenceModel> model)
      : model_(std::move(model)) {}

  InferenceRunner(InferenceRunner&&) = default;
  InferenceRunner& operator=(InferenceRunner&&) = default;

  InferenceRunner(const InferenceRunner&) = delete;
  InferenceRunner& operator=(const InferenceRunner&) = delete;

  static absl::StatusOr<InferenceRunner> BuildModel(
      const local_inference::SessionInferenceConfiguration&
          inference_configuration);
  const local_inference::SessionInferenceConfiguration&
  GetInferenceConfiguration() const;

  // Runs inference on the given input. The input is expected to contain the
  // columns required by the inference task. The output column produced by
  // the inference model will be added to the input.
  //
  // Inference tasks are applied in order, and each task sees the input as
  // modified by the tasks before it. A task whose output expands the row count
  // therefore also expands the input seen by subsequent tasks.
  absl::Status RunInference(Input& input);

 private:
  std::unique_ptr<local_inference::InferenceModel> model_;
  inference::InferencePromptProcessor prompt_processor_;
  inference::InferenceOutputProcessor output_processor_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_INFERENCE_RUNNER_H_
