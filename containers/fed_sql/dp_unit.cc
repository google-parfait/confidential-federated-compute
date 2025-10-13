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
#include "containers/fed_sql/dp_unit.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "containers/fed_sql/interval_set.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute::fed_sql {

namespace {
using confidential_federated_compute::sql::Input;
using confidential_federated_compute::sql::RowLocation;
using confidential_federated_compute::sql::RowSet;
using confidential_federated_compute::sql::SqliteAdapter;
using tensorflow_federated::aggregation::CheckpointParser;
using tensorflow_federated::aggregation::Tensor;
}  // namespace

absl::StatusOr<std::vector<absl::Status>>
DpUnitProcessor::CommitRowsGroupingByDpUnit(
    std::vector<Input>&& uncommitted_inputs,
    std::vector<RowLocation>&& row_dp_unit_index) {
  std::vector<absl::Status> ignored_errors;
  // TODO: Iterate over uncommitted partition keys and ensure they're in the
  // specified fedsql::Interval range.
  std::sort(row_dp_unit_index.begin(), row_dp_unit_index.end());
  for (auto it = row_dp_unit_index.begin(); it != row_dp_unit_index.end();) {
    const auto& dp_unit_hash = it->dp_unit_hash;
    // Get a span of rows that belong to the same DP unit.
    auto end_of_range = std::find_if(
        it, row_dp_unit_index.end(), [&dp_unit_hash](const auto& other) {
          return dp_unit_hash != other.dp_unit_hash;
        });
    absl::Span<const RowLocation> dp_unit_span(&*it,
                                               std::distance(it, end_of_range));
    it = end_of_range;  // Move to the next DP unit.
    FCP_ASSIGN_OR_RETURN(RowSet row_set,
                         RowSet::Create(dp_unit_span, uncommitted_inputs));
    absl::StatusOr<std::vector<Tensor>> sql_result =
        ExecuteClientQuery(sql_configuration_, row_set);
    // Errors from the SQL query itself (eg. division by zero, invalid column
    // references, etc.) are ignored.
    if (!sql_result.ok()) {
      ignored_errors.push_back(sql_result.status());
      continue;
    }
    InMemoryCheckpointParser parser(*std::move(sql_result));

    // In case of an error with Accumulate, the session is terminated, since we
    // can't guarantee that the aggregator is in a valid state. If this
    // changes, consider changing this logic to no longer return an error.
    absl::Status accumulate_status = aggregator_.Accumulate(parser);
    if (!accumulate_status.ok()) {
      if (absl::IsNotFound(accumulate_status)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Failed to accumulate SQL query results: ",
                         accumulate_status.message()));
      }
      return accumulate_status;
    }
  }
  return ignored_errors;
}

}  // namespace confidential_federated_compute::fed_sql
