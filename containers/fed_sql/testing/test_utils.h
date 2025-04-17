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
// Helper functions and classes for FedSQL unit tests.
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_TEST_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_TEST_UTILS_H_

#include <string>

namespace confidential_federated_compute::fed_sql::testing {

std::string BuildFedSqlGroupByCheckpoint(
    std::initializer_list<uint64_t> key_col_values,
    std::initializer_list<uint64_t> val_col_values,
    const std::string& key_col_name = "key",
    const std::string& val_col_name = "val");

}  // namespace confidential_federated_compute::fed_sql::testing

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_TEST_UTILS_H_
