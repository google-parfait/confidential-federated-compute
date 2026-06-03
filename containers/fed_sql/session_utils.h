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

#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

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
            std::optional<fcp::confidentialcompute::InferenceConfiguration>
                inference_configuration = std::nullopt);

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_SESSION_UTILS_H_
