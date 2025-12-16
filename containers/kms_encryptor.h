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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_KMS_ENCRYPTOR_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_KMS_ENCRYPTOR_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute {

// Re-encryption context used by the Session::Context implementation
// to encrypt emitted results.
class KmsEncryptor {
 public:
  // The encrypted intermediate or final result.
  struct EncryptedResult {
    std::string ciphertext;
    fcp::confidentialcompute::BlobMetadata metadata;
    // The configuration for the final result. This is not populated for
    // intermediate results.
    std::optional<fcp::confidentialcompute::FinalizeResponse> finalize_response;
  };

  KmsEncryptor(std::vector<std::string> reencryption_keys,
               std::string reencryption_policy_hash)
      : reencryption_keys_(std::move(reencryption_keys)),
        reencryption_policy_hash_(std::move(reencryption_policy_hash)) {}

  absl::StatusOr<EncryptedResult> EncryptIntermediateResult(
      int reencryption_key_index, absl::string_view plaintext,
      absl::string_view blob_id) const;

  // The reencryption keys used to re-encrypt the intermediate and final blobs.
  const std::vector<std::string>& reencryption_keys() const {
    return reencryption_keys_;
  }
  // The policy hash used to re-encrypt the intermediate and final blobs with.
  const std::string& reencryption_policy_hash() const {
    return reencryption_policy_hash_;
  }

 private:
  absl::StatusOr<absl::string_view> GetReencryptionKey(
      int reencryption_key_index) const;
  absl::StatusOr<std::string> CreateAssociatedData(
      absl::string_view reencryption_key, absl::string_view blob_id) const;
  fcp::confidentialcompute::BlobMetadata CreateMetadata(
      const fcp::confidential_compute::EncryptMessageResult& encrypted_message,
      absl::string_view blob_id, absl::string_view associated_data) const;

  std::vector<std::string> reencryption_keys_;
  std::string reencryption_policy_hash_;
  const fcp::confidential_compute::MessageEncryptor message_encryptor_;
};

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_KMS_ENCRYPTOR_H_
