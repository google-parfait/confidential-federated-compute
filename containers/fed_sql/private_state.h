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
#include <string>

#include "containers/fed_sql/budget.h"

namespace confidential_federated_compute::fed_sql {

// FedSql private state that is persisted via KMS.
struct PrivateState {
  PrivateState(std::string initial_state, uint32_t default_budget)
      : initial_state(std::move(initial_state)), budget(default_budget) {}

  // The initial serialized budget received from KMS.
  std::string initial_state;
  // The current budget.
  Budget budget;
};

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_PRIVATE_STATE_H_
