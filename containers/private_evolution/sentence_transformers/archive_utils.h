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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_ARCHIVE_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_ARCHIVE_UTILS_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace confidential_federated_compute::sentence_transformers {

inline constexpr absl::string_view kTmpRoot = "/tmp";

// Extract all files inside the archive into temp directory.
// Returns the full path to the top level directory after extraction.
absl::StatusOr<std::string> ExtractAll(absl::string_view zip_file_path,
                                       absl::string_view parent = kTmpRoot);

}  // namespace confidential_federated_compute::sentence_transformers

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_ARCHIVE_UTILS_H_
