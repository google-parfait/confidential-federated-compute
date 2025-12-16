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

#include "containers/kms_encryptor.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"

namespace confidential_federated_compute {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;

absl::StatusOr<absl::string_view> KmsEncryptor::GetReencryptionKey(
    int reencryption_key_index) const {
  if (reencryption_key_index < 0 ||
      reencryption_key_index >= reencryption_keys_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid reencryption index ", reencryption_key_index));
  }
  return reencryption_keys_[reencryption_key_index];
}

absl::StatusOr<std::string> KmsEncryptor::CreateAssociatedData(
    absl::string_view reencryption_key, absl::string_view blob_id) const {
  FCP_ASSIGN_OR_RETURN(OkpKey okp_key, OkpKey::Decode(reencryption_key));

  BlobHeader header;
  header.set_blob_id(blob_id);
  header.set_key_id(okp_key.key_id);
  header.set_access_policy_sha256(reencryption_policy_hash_);
  return header.SerializeAsString();
}

BlobMetadata KmsEncryptor::CreateMetadata(
    const EncryptMessageResult& encrypted_message, absl::string_view blob_id,
    absl::string_view associated_data) const {
  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(encrypted_message.ciphertext.size());

  BlobMetadata::HpkePlusAeadMetadata* hpke_plus_aead_metadata =
      metadata.mutable_hpke_plus_aead_data();
  hpke_plus_aead_metadata->set_ciphertext_associated_data(associated_data);
  hpke_plus_aead_metadata->set_encrypted_symmetric_key(
      encrypted_message.encrypted_symmetric_key);
  hpke_plus_aead_metadata->set_encapsulated_public_key(
      encrypted_message.encapped_key);
  hpke_plus_aead_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header(associated_data);
  hpke_plus_aead_metadata->set_blob_id(blob_id);
  return metadata;
}

absl::StatusOr<KmsEncryptor::EncryptedResult>
KmsEncryptor::EncryptIntermediateResult(int reencryption_key_index,
                                        absl::string_view plaintext,
                                        absl::string_view blob_id) const {
  FCP_ASSIGN_OR_RETURN(absl::string_view reencryption_key,
                       GetReencryptionKey(reencryption_key_index));
  FCP_ASSIGN_OR_RETURN(std::string associated_data,
                       CreateAssociatedData(reencryption_key, blob_id));
  FCP_ASSIGN_OR_RETURN(
      EncryptMessageResult encrypted_message,
      message_encryptor_.Encrypt(plaintext, reencryption_key, associated_data));

  BlobMetadata metadata =
      CreateMetadata(encrypted_message, blob_id, associated_data);
  return EncryptedResult{.ciphertext = std::move(encrypted_message.ciphertext),
                         .metadata = std::move(metadata)};
}

}  // namespace confidential_federated_compute
