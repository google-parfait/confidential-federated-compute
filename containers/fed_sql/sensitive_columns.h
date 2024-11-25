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

// This file contains functions for handling sensitive columns.
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_SENSITIVE_COLUMNS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_SENSITIVE_COLUMNS_H_

#include <string>

#include "absl/status/status.h"
#include "containers/sql/sqlite_adapter.h"

namespace confidential_federated_compute::fed_sql {

// Update the values of any columns with the "SENSITIVE_" prefix by hashing them
// with `key` (using HMAC-SHA256). Columns with the "SENSITIVE_" prefix are
// required to be either the STRING or BYTES type.
absl::Status HashSensitiveColumns(
    std::vector<confidential_federated_compute::sql::TensorColumn>& contents,
    absl::string_view key);

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_SENSITIVE_COLUMNS_H_
