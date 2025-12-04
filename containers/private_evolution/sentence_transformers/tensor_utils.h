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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_TENSOR_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_TENSOR_UTILS_H_

#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sentence_transformers {

absl::StatusOr<tensorflow_federated::aggregation::Tensor> CreateEmbeddingTensor(
    std::vector<std::vector<float>> embeddings);

}  // namespace confidential_federated_compute::sentence_transformers

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_TENSOR_UTILS_H_
