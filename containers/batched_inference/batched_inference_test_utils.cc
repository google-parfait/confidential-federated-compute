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

#include "containers/batched_inference/batched_inference_test_utils.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "google/protobuf/any.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"

namespace confidential_federated_compute::batched_inference::testing {

using ::fcp::confidentialcompute::InferenceConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

InferenceConfiguration GetInferenceConfigForTest() {
  InferenceConfiguration inference_config;
  auto* task = inference_config.add_inference_task();
  task->mutable_column_config()->add_input_column_names("transcript");
  task->mutable_column_config()->set_output_column_name("topic");
  auto* prompt = task->mutable_prompt();
  prompt->set_prompt_template("Hello, {transcript}");
  inference_config.mutable_runtime_config()->set_max_prompt_size(1000);
  inference_config.mutable_runtime_config()->set_max_batch_size(3);
  return inference_config;
}

void AddInitConfigForTest(StreamInitializeRequest* init_request) {
  InferenceInitializeConfiguration init_config;
  *init_config.mutable_inference_config() = GetInferenceConfigForTest();
  *init_request->mutable_initialize_request()->mutable_configuration() = Any();
  init_request->mutable_initialize_request()->mutable_configuration()->PackFrom(
      init_config);
}

namespace {

void AddVecToBuilderOrDie(
    tensorflow_federated::aggregation::CheckpointBuilder* builder,
    std::string name, std::vector<std::string> vec) {
  std::vector<absl::string_view> views(vec.begin(), vec.end());
  auto tensor_data =
      std::make_unique<MutableVectorData<absl::string_view>>(std::move(views));
  auto tensor_or = Tensor::Create(
      DataType::DT_STRING, TensorShape({static_cast<int64_t>(vec.size())}),
      std::move(tensor_data), name);
  ABSL_CHECK(tensor_or.ok());
  builder->Add(name, std::move(*tensor_or));
}

}  // namespace

std::string GetPrivateInferenceInputCheckpointForTest(
    std::vector<std::string> prompts) {
  auto builder = FederatedComputeCheckpointBuilderFactory().Create();
  AddVecToBuilderOrDie(builder.get(), "transcript", prompts);
  return std::string(builder->Build().value());
}

std::string GetPrivateInferenceOutputCheckpointForTest(
    std::vector<std::string> prompts, std::vector<std::string> results) {
  auto builder = FederatedComputeCheckpointBuilderFactory().Create();
  AddVecToBuilderOrDie(builder.get(), "transcript", prompts);
  AddVecToBuilderOrDie(builder.get(), "topic", results);
  return std::string(builder->Build().value());
}

}  // namespace confidential_federated_compute::batched_inference::testing
