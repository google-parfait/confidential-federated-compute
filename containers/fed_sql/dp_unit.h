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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_DP_UNIT_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_DP_UNIT_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/civil_time.h"
#include "absl/types/span.h"
#include "containers/fed_sql/interval_set.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/sql/row_set.h"
#include "containers/sql/row_view.h"
#include "fcp/protos/confidentialcompute/windowing_schedule.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute::fed_sql {

// Holds parameters related to the differential privacy unit.
struct DpUnitParameters {
  // The windowing schedule for the DP unit.
  fcp::confidentialcompute::WindowingSchedule windowing_schedule;
  // The column names that define the DP unit (not including the event time
  // or privacy id columns).
  std::vector<std::string> column_names;
};

// Processes inputs so differential privacy is applied at a specified unit.
class DpUnitProcessor {
 public:
  // Creates a DpUnitProcessor.
  static absl::StatusOr<std::unique_ptr<DpUnitProcessor>> Create(
      const SqlConfiguration& sql_configuration,
      const DpUnitParameters& dp_unit_parameters,
      tensorflow_federated::aggregation::CheckpointAggregator* aggregator);

  DpUnitProcessor(
      SqlConfiguration sql_configuration, DpUnitParameters dp_unit_parameters,
      tensorflow_federated::aggregation::CheckpointAggregator* aggregator)
      : sql_configuration_(std::move(sql_configuration)),
        dp_unit_parameters_(std::move(dp_unit_parameters)),
        aggregator_(aggregator) {};

  // Computes the start of the DP time unit for the given civil time.
  absl::StatusOr<absl::CivilSecond> ComputeDPTimeUnit(
      absl::CivilSecond start_civil_time);

  /// Returns a hash of the DP time unit and DP column values. This is used to
  // group rows by DP unit.
  absl::StatusOr<uint64_t> ComputeDPUnitHash(
      int64_t privacy_id, absl::CivilSecond dp_time_unit,
      confidential_federated_compute::sql::RowView row_view,
      absl::Span<const int64_t> dp_indices);

  // Executes the SQL query on each DP unit and accumulates the results using
  // the provided aggregator. Returns any ignored errors. Not thread safe.
  absl::StatusOr<std::vector<absl::Status>> CommitRowsGroupingByDpUnit(
      std::vector<confidential_federated_compute::sql::Input>&&
          uncommitted_inputs,
      std::vector<confidential_federated_compute::sql::RowLocation>&&
          row_dp_unit_index);

 private:
  SqlConfiguration sql_configuration_;
  DpUnitParameters dp_unit_parameters_;
  tensorflow_federated::aggregation::CheckpointAggregator* aggregator_;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_DP_UNIT_H_
