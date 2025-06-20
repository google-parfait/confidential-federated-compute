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

#include "containers/program_executor_tee/program_context/cc/generate_checkpoint.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"

namespace confidential_federated_compute::program_executor_tee {

using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

std::string BuildClientCheckpointFromInts(std::vector<int> input_values,
                                          std::string tensor_name) {
  absl::StatusOr<Tensor> t =
      Tensor::Create(DataType::DT_INT32,
                     TensorShape({static_cast<int32_t>(input_values.size())}),
                     std::make_unique<MutableVectorData<int32_t>>(
                         input_values.begin(), input_values.end()));
  CHECK_OK(t);
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();
  CHECK_OK(ckpt_builder->Add(tensor_name, *t));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint);
  return std::string(*checkpoint);
}

}  // namespace confidential_federated_compute::program_executor_tee