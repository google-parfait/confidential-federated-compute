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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_JSON_PARSING_UTILS_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_JSON_PARSING_UTILS_H

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json.hpp"

namespace confidential_federated_compute::gcp {

// Helper to extract a potentially nested string claim from a parsed JSON object
// using a vector of keys representing the path.
absl::StatusOr<std::string> GetStringClaimFromPath(
    const nlohmann::json& payload_json,
    const std::vector<absl::string_view>& path);

// Helper to extract a potentially nested list of strings from a parsed JSON
// object. Returns an empty vector if the path does not exist. Returns an error
// if the path exists but is not an array, or contains non-string elements.
absl::StatusOr<std::vector<std::string>> GetStringListClaimFromPath(
    const nlohmann::json& payload_json,
    const std::vector<absl::string_view>& path);

}  // namespace confidential_federated_compute::gcp

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_JSON_PARSING_UTILS_H
