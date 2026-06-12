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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_UTILS_H_

#include <string>

#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::fns {

// Returns the blob ID from the BlobMetadata if it exists, otherwise returns an
// empty string. Shared by DoFn and PObjectMapFn.
std::string GetBlobId(const fcp::confidentialcompute::BlobMetadata& metadata);

}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_UTILS_H_
