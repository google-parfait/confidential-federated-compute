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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_PRIVATE_STATE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_PRIVATE_STATE_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/cord.h"
#include "containers/fed_sql/budget.h"
#include "containers/fed_sql/range_tracker.h"
#include "range_tracker.h"

namespace confidential_federated_compute::fed_sql {

// FedSql private state shared from ConfidentialTransformServer to all
// the sessions created by it. This combines state that is persisted in KMS
// with the state received during the container transform initialization.
struct PrivateState {
  PrivateState(std::optional<std::string> initial_state,
               std::optional<uint32_t> default_budget,
               std::optional<RangeTracker> consumed_tracker = std::nullopt,
               absl::Cord autotuning_data_to_release = absl::Cord{})
      : initial_state(std::move(initial_state)),
        budget(default_budget),
        consumed_tracker(std::move(consumed_tracker).value_or(RangeTracker())),
        autotuning_data_to_release(std::move(autotuning_data_to_release)) {}

  // The initial serialized budget received from KMS.
  std::optional<std::string> initial_state;
  // The current budget.
  Budget budget;
  // The part of the budget that is initially consumed by auto-tuning.
  RangeTracker consumed_tracker;
  // Optional auto-tuning data to be released. If there is no data to release
  // this Cord remains empty.
  absl::Cord autotuning_data_to_release;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_PRIVATE_STATE_H_
