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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_CHECKPOINT_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_CHECKPOINT_UTILS_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute {

// Extracts the Privacy ID from a client upload checkpoint.
// Validates that the `confidential_compute_privacy_id` tensor exists, is a
// string tensor, and is a scalar.
absl::StatusOr<std::string> GetPrivacyId(
    tensorflow_federated::aggregation::CheckpointParser& parser);

// Extracts the event time column from a client upload checkpoint.
// The tensor fully qualified name is
// `<on_device_query_name>/confidential_compute_event_time`. Validates that the
// tensor exists, is a string tensor, and has exactly one dimension.
absl::StatusOr<std::vector<std::string>> GetEventTime(
    tensorflow_federated::aggregation::CheckpointParser& parser,
    absl::string_view on_device_query_name);

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_CHECKPOINT_UTILS_H_
