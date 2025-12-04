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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_KMS_HELPER_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_KMS_HELPER_H

#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "cc/crypto/signing_key.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"

namespace confidential_federated_compute::program_executor_tee {

constexpr size_t kBlobIdSize = 16;

absl::Status CreateWriteRequestForRelease(
    fcp::confidentialcompute::outgoing::WriteRequest* write_request,
    oak::crypto::SigningKeyHandle& signing_key,
    absl::string_view encryption_key, std::string key, std::string data,
    std::string access_policy_hash,
    std::optional<std::string> src_state = std::nullopt,
    std::string dst_state = "");

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_KMS_HELPER_H
