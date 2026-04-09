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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_DATA_INGRESS_SQL_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_DATA_INGRESS_SQL_UTILS_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "containers/sql/row_set.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute::sql_data_ingress {

struct SqlConfiguration {
  std::string query;
  fcp::confidentialcompute::TableSchema input_schema;
  google::protobuf::RepeatedPtrField<fcp::confidentialcompute::ColumnSchema>
      output_columns;
};

absl::StatusOr<std::vector<tensorflow_federated::aggregation::Tensor>>
Deserialize(const fcp::confidentialcompute::TableSchema& table_schema,
            tensorflow_federated::aggregation::CheckpointParser* checkpoint);

absl::StatusOr<std::vector<tensorflow_federated::aggregation::Tensor>>
ExecuteClientQuery(const SqlConfiguration& configuration, sql::RowSet rows);

std::vector<sql::RowLocation> CreateRowLocationsForAllRows(size_t num_rows);

}  // namespace confidential_federated_compute::sql_data_ingress

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_DATA_INGRESS_SQL_UTILS_H_
