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
#include "tensor_utils.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace confidential_federated_compute::sentence_transformers {
namespace {
using ::tensorflow_federated::aggregation::DT_FLOAT;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorData;
using ::tensorflow_federated::aggregation::TensorShape;

class NestedFloatVectorData : public TensorData {
 public:
  explicit NestedFloatVectorData(std::vector<std::vector<float>> data) {
    for (auto& inner_vector : data) {
      for (auto& element : inner_vector) {
        flattened_data_.push_back(std::move(element));
      }
    }
  }
  const void* data() const override { return flattened_data_.data(); }
  size_t byte_size() const override {
    return flattened_data_.size() * sizeof(float);
  }

 private:
  std::vector<float> flattened_data_;
};

bool VerifyInnerVectorsHaveSameSize(
    const std::vector<std::vector<float>>& nested_vector, int32_t size) {
  for (const auto& inner_vector : nested_vector) {
    if (inner_vector.size() != size) {
      return false;
    }
  }
  return true;
}

}  // anonymous namespace

absl::StatusOr<Tensor> CreateEmbeddingTensor(
    std::vector<std::vector<float>> embeddings) {
  if (embeddings.empty()) {
    return absl::InvalidArgumentError("Missing embedding vectors.");
  }
  int32_t dim_0 = embeddings.size();
  int32_t dim_1 = embeddings[0].size();
  if (!VerifyInnerVectorsHaveSameSize(embeddings, dim_1)) {
    return absl::InvalidArgumentError(
        "Embeddings are having different shapes.");
  }
  return Tensor::Create(
      DT_FLOAT, TensorShape({dim_0, dim_1}),
      std::make_unique<NestedFloatVectorData>(std::move(embeddings)));
}

}  // namespace confidential_federated_compute::sentence_transformers