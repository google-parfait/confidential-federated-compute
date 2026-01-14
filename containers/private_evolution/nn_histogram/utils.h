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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_NN_HISTOGRAM_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_NN_HISTOGRAM_UTILS_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"

namespace confidential_federated_compute::nn_histogram {

// Read the file into a vector of Embedding protos.  Returns error if failed to
// read the file or the file content is not in the expected format.
absl::StatusOr<std::vector<fcp::confidentialcompute::Embedding>> ReadRecords(
    absl::string_view file_path);

// Calculate the cartesian distance between input and every synthetic data
// embeddings, return the synthetic data index with the smallest distance.
// Returns an error status if the vector size of input and synthetic data are
// different.
absl::StatusOr<int32_t> FindNearestNeighbor(
    absl::Span<const float> input_embedding,
    const std::vector<fcp::confidentialcompute::Embedding>&
        synthetic_data_embeddings);

}  // namespace confidential_federated_compute::nn_histogram

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_NN_HISTOGRAM_UTILS_H_
