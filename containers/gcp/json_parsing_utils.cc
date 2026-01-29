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

#include "json_parsing_utils.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace confidential_federated_compute::gcp {

absl::StatusOr<std::string> GetStringClaimFromPath(
    const nlohmann::json& payload_json,
    const std::vector<absl::string_view>& path) {
  if (path.empty()) {
    return absl::InvalidArgumentError("Path cannot be empty.");
  }

  const nlohmann::json* current_node = &payload_json;
  for (size_t i = 0; i < path.size(); ++i) {
    absl::string_view key = path[i];
    // Check if the current node is an object before attempting lookup.
    if (!current_node->is_object()) {
      return absl::NotFoundError(absl::StrCat(
          "Path traversal failed: Expected object at step ", i, " for key '",
          key, "', but found ", current_node->type_name()));
    }
    // Find the key in the current object.
    auto it = current_node->find(key);
    if (it == current_node->end()) {
      return absl::NotFoundError(absl::StrCat(
          "Claim path not found: Missing key '", key, "' at step ", i));
    }
    // Move to the next node in the path.
    current_node = &(*it);
  }

  // Ensure the final node is a string.
  if (!current_node->is_string()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Claim path found, but final value is not a string (found type: ",
        current_node->type_name(), ") for key '", path.back(), "'"));
  }

  return current_node->get<std::string>();
}

absl::StatusOr<std::vector<std::string>> GetStringListClaimFromPath(
    const nlohmann::json& payload_json,
    const std::vector<absl::string_view>& path) {
  if (path.empty()) {
    return absl::InvalidArgumentError("Path cannot be empty.");
  }

  const nlohmann::json* current_node = &payload_json;
  // Traverse to the parent of the final key.
  for (size_t i = 0; i < path.size(); ++i) {
    absl::string_view key = path[i];
    if (!current_node->is_object()) {
      // If any intermediate node is missing or not an object, treat as not
      // found (empty list).
      return std::vector<std::string>{};
    }
    auto it = current_node->find(key);
    if (it == current_node->end()) {
      return std::vector<std::string>{};
    }
    current_node = &(*it);
  }

  // The final node must be an array.
  if (!current_node->is_array()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Claim path found, but value is not an array (found type: ",
        current_node->type_name(), ") for key '", path.back(), "'"));
  }

  std::vector<std::string> result;
  for (const auto& element : *current_node) {
    if (!element.is_string()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Array at '", path.back(),
          "' contains non-string elements (found type: ", element.type_name(),
          ")"));
    }
    result.push_back(element.get<std::string>());
  }

  return result;
}

}  // namespace confidential_federated_compute::gcp
