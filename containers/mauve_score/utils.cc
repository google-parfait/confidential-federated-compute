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

#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/records/record_reader.h"

namespace confidential_federated_compute::mauve_score {

using ::fcp::confidentialcompute::Embedding;

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

}  // namespace confidential_federated_compute::mauve_score
