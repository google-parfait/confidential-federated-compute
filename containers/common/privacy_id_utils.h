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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_PRIVACY_ID_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_PRIVACY_ID_UTILS_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "containers/big_endian.h"
#include "fcp/base/digest.h"

namespace confidential_federated_compute {

// Computes the upper 64 bits of the SHA-256 hash of the given privacy ID.
inline uint64_t ComputeUpper64HashedPrivacyId(absl::string_view privacy_id) {
  return LoadBigEndian<uint64_t>(fcp::ComputeSHA256(std::string(privacy_id)));
}

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_PRIVACY_ID_UTILS_H_
