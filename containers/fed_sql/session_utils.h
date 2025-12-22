// Copyright 2024 Google LLC.
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
// Helper functions and classes for FedSQL unit tests.
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_SESSION_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_SESSION_UTILS_H_

#include <string>
#include <tuple>

#include "absl/container/flat_hash_map.h"
#include "containers/fed_sql/inference_model.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gemma/gemma.h"
#include "gmock/gmock.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"

namespace confidential_federated_compute::fed_sql {

// Deserialize a FedSQL checkpoint.
// The columns are specified in the `table_schema`.
// If an `inference_configuration`, then the checkpoint is expected to contain
// the input columns specified in the `inference_configuration`. The output
// columns specified in the `inference_configuration` should be part of the
// `table_schema`, but will be generated at inference time so they are not
// expected to be in the checkpoint.
// TODO: Factor out the inference-specific logic and move this helper to the
// containers/sql folder.
absl::StatusOr<std::vector<tensorflow_federated::aggregation::Tensor>>
Deserialize(const fcp::confidentialcompute::TableSchema& table_schema,
            tensorflow_federated::aggregation::CheckpointParser* checkpoint,
            std::optional<SessionInferenceConfiguration>
                inference_configuration = std::nullopt);

// A simple pass-through CheckpointParser.
class InMemoryCheckpointParser
    : public tensorflow_federated::aggregation::CheckpointParser {
 public:
  explicit InMemoryCheckpointParser(
      std::vector<tensorflow_federated::aggregation::Tensor> columns) {
    for (auto& column : columns) {
      tensors_[column.name()] = std::move(column);
    }
  }

  absl::StatusOr<tensorflow_federated::aggregation::Tensor> GetTensor(
      const std::string& name) override {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      return absl::NotFoundError(absl::StrCat("Tensor not found: ", name));
    }
    return std::move(it->second);
  }

 private:
  absl::flat_hash_map<std::string, tensorflow_federated::aggregation::Tensor>
      tensors_;
};

// Creates a RowLocation for each row in `columns`. All columns must have the
// same number of rows.
// Helper function for creating RowLocations for the contents of a single
// input. This will be used while custom DP units is still in development,
// however, eventually all uses will group rows by DP unit.
std::vector<sql::RowLocation> CreateRowLocationsForAllRows(size_t num_rows);

// Configuration of the per-client SQL query step.
struct SqlConfiguration {
  std::string query;
  fcp::confidentialcompute::TableSchema input_schema;
  google::protobuf::RepeatedPtrField<fcp::confidentialcompute::ColumnSchema>
      output_columns;
};

// Executes the given SQL query on a set of rows.
absl::StatusOr<std::vector<tensorflow_federated::aggregation::Tensor>>
ExecuteClientQuery(const SqlConfiguration& configuration, sql::RowSet rows);

}  // namespace confidential_federated_compute::fed_sql
#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_SESSION_UTILS_H_
