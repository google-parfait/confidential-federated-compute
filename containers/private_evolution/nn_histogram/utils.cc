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
#include "utils.h"

#include <filesystem>
#include <limits>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"
#include "google/protobuf/repeated_field.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/records/record_reader.h"

namespace confidential_federated_compute::nn_histogram {
namespace {
using ::fcp::confidentialcompute::Embedding;
using ::google::protobuf::RepeatedField;

double SquaredCartesianDistance(absl::Span<const float> input_emb,
                                const RepeatedField<float>& syn_data_emb) {
  double distance = 0;
  for (int i = 0; i < input_emb.size(); i++) {
    double diff = static_cast<double>(input_emb.at(i)) - syn_data_emb[i];
    distance += diff * diff;
  }
  return distance;
}

}  // namespace

absl::StatusOr<std::vector<Embedding>> ReadRecords(
    absl::string_view file_path) {
  riegeli::RecordReader reader(
      riegeli::Maker<riegeli::FdReader>(std::string(file_path)));
  std::vector<Embedding> synthetic_data_embeddings;
  Embedding record;
  while (reader.ReadRecord(record)) {
    synthetic_data_embeddings.push_back(record);
  }
  if (!reader.Close()) {
    return reader.status();
  }
  return synthetic_data_embeddings;
}

absl::StatusOr<int32_t> FindNearestNeighbor(
    absl::Span<const float> input_embedding,
    const std::vector<Embedding>& synthetic_data_embeddings) {
  size_t s = input_embedding.size();
  if (synthetic_data_embeddings.empty()) {
    return absl::InvalidArgumentError("Missing synthetic data embeddings.");
  }

  double min_distance = std::numeric_limits<double>::max();
  int32_t index = -1;
  for (const auto& embedding : synthetic_data_embeddings) {
    if (embedding.values().size() != s) {
      return absl::InvalidArgumentError(
          "Synthetic data embedding has different dimensions than the input "
          "data embedding.");
    }
    auto distance =
        SquaredCartesianDistance(input_embedding, embedding.values());
    if (distance < min_distance) {
      min_distance = distance;
      index = embedding.index();
    }
  }
  return index;
}

}  // namespace confidential_federated_compute::nn_histogram
