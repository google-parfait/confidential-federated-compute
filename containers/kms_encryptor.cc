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
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::google::protobuf::Any;

namespace {

// Builds BlobMetadata with the common HpkePlusAeadMetadata fields shared by
// both metadata creation paths.
BlobMetadata BuildCommonMetadata(const EncryptMessageResult& encrypted_message,
                                 absl::string_view blob_id,
                                 absl::string_view ciphertext_associated_data) {
  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(encrypted_message.ciphertext.size());

  BlobMetadata::HpkePlusAeadMetadata* hpke_plus_aead_metadata =
      metadata.mutable_hpke_plus_aead_data();
  hpke_plus_aead_metadata->set_ciphertext_associated_data(
      ciphertext_associated_data);
  hpke_plus_aead_metadata->set_encrypted_symmetric_key(
      encrypted_message.encrypted_symmetric_key);
  hpke_plus_aead_metadata->set_encapsulated_public_key(
      encrypted_message.encapped_key);
  hpke_plus_aead_metadata->set_blob_id(blob_id);
  return metadata;
}

}  // namespace

absl::StatusOr<absl::string_view> KmsEncryptor::GetReencryptionKey(
    int reencryption_key_index) const {
  if (reencryption_key_index < 0 ||
      reencryption_key_index >= reencryption_keys_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid reencryption index ", reencryption_key_index));
  }
  return reencryption_keys_[reencryption_key_index];
}

absl::StatusOr<std::string> KmsEncryptor::CreateSerializedBlobHeader(
    absl::string_view reencryption_key, absl::string_view blob_id) const {
  ABSL_ASSIGN_OR_RETURN(OkpKey okp_key, OkpKey::Decode(reencryption_key));

  BlobHeader header;
  header.set_blob_id(blob_id);
  header.set_key_id(okp_key.key_id);
  return header.SerializeAsString();
}

// TODO: Remove the legacy codepath that uses record_header once we've verified
// that this won't break any containers.
BlobMetadata KmsEncryptor::CreateMetadataWithBlobHeader(
    const EncryptMessageResult& encrypted_message, absl::string_view blob_id,
    absl::string_view blob_header) const {
  BlobMetadata metadata =
      BuildCommonMetadata(encrypted_message, blob_id, blob_header);

  // blob_header is a serialized BlobHeader which contains key_id inside it.
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_kms_symmetric_key_associated_data()
      ->set_record_header(blob_header);

  return metadata;
}

BlobMetadata KmsEncryptor::CreateMetadataWithAssociatedMetadata(
    const EncryptMessageResult& encrypted_message, absl::string_view blob_id,
    absl::string_view key_id, Any associated_metadata) const {
  BlobMetadata metadata = BuildCommonMetadata(encrypted_message, blob_id,
                                              associated_metadata.value());

  // Pack AssociatedMetadata into KmsAssociatedData.associated_metadata. The
  // deprecated record_header field is intentionally left empty.
  *metadata.mutable_hpke_plus_aead_data()
       ->mutable_kms_symmetric_key_associated_data()
       ->mutable_associated_metadata() = std::move(associated_metadata);
  // Set key_id directly on HpkePlusAeadMetadata.
  metadata.mutable_hpke_plus_aead_data()->set_key_id(key_id);

  return metadata;
}

absl::StatusOr<KmsEncryptor::EncryptedResult>
KmsEncryptor::EncryptIntermediateResult(int reencryption_key_index,
                                        absl::string_view plaintext,
                                        absl::string_view blob_id) const {
  ABSL_ASSIGN_OR_RETURN(absl::string_view reencryption_key,
                        GetReencryptionKey(reencryption_key_index));
  ABSL_ASSIGN_OR_RETURN(std::string associated_data,
                        CreateSerializedBlobHeader(reencryption_key, blob_id));
  ABSL_ASSIGN_OR_RETURN(
      EncryptMessageResult encrypted_message,
      message_encryptor_.Encrypt(plaintext, reencryption_key, associated_data));

  BlobMetadata metadata =
      CreateMetadataWithBlobHeader(encrypted_message, blob_id, associated_data);
  return EncryptedResult{.ciphertext = std::move(encrypted_message.ciphertext),
                         .metadata = std::move(metadata)};
}

absl::StatusOr<KmsEncryptor::EncryptedResult>
KmsEncryptor::EncryptIntermediateResult(
    int reencryption_key_index, absl::string_view plaintext,
    absl::string_view blob_id,
    const fcp::confidentialcompute::AssociatedMetadata& associated_metadata)
    const {
  ABSL_ASSIGN_OR_RETURN(absl::string_view reencryption_key,
                        GetReencryptionKey(reencryption_key_index));
  ABSL_ASSIGN_OR_RETURN(OkpKey okp_key, OkpKey::Decode(reencryption_key));
  Any associated_metadata_any;
  associated_metadata_any.PackFrom(associated_metadata);
  ABSL_ASSIGN_OR_RETURN(
      EncryptMessageResult encrypted_message,
      message_encryptor_.Encrypt(plaintext, reencryption_key,
                                 associated_metadata_any.value()));

  BlobMetadata metadata = CreateMetadataWithAssociatedMetadata(
      encrypted_message, blob_id, okp_key.key_id,
      std::move(associated_metadata_any));
  return EncryptedResult{.ciphertext = std::move(encrypted_message.ciphertext),
                         .metadata = std::move(metadata)};
}

absl::StatusOr<KmsEncryptor::EncryptedResult>
KmsEncryptor::EncryptReleasableResult(
    int reencryption_key_index, absl::string_view plaintext,
    absl::string_view blob_id, std::optional<absl::string_view> src_state,
    absl::string_view dst_state) const {
  ABSL_ASSIGN_OR_RETURN(absl::string_view reencryption_key,
                        GetReencryptionKey(reencryption_key_index));
  ABSL_ASSIGN_OR_RETURN(std::string associated_data,
                        CreateSerializedBlobHeader(reencryption_key, blob_id));
  ABSL_ASSIGN_OR_RETURN(
      EncryptMessageResult encrypted_message,
      message_encryptor_.EncryptForRelease(
          plaintext, reencryption_key, associated_data, src_state, dst_state,
          [this](absl::string_view message) -> absl::StatusOr<std::string> {
            ABSL_ASSIGN_OR_RETURN(auto signature,
                                  signing_key_handle_->Sign(message));
            return std::move(*signature.mutable_signature());
          }));

  BlobMetadata metadata =
      CreateMetadataWithBlobHeader(encrypted_message, blob_id, associated_data);

  return EncryptedResult{
      .ciphertext = std::move(encrypted_message.ciphertext),
      .metadata = std::move(metadata),
      .release_token = std::move(encrypted_message.release_token)};
}

}  // namespace confidential_federated_compute
