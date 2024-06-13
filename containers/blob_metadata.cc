// Copyright 2024 Google LLC.
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
#include "containers/blob_metadata.h"

#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"

namespace confidential_federated_compute {

using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::Record;

absl::StatusOr<BlobMetadata> EarliestExpirationTimeMetadata(
    const BlobMetadata& metadata, const BlobMetadata& other) {
  absl::Time metadata_time = absl::InfiniteFuture();
  absl::Time other_time = absl::InfiniteFuture();
  if (metadata.has_hpke_plus_aead_data()) {
    FCP_ASSIGN_OR_RETURN(
        OkpCwt metadata_cwt,
        OkpCwt::Decode(metadata.hpke_plus_aead_data()
                           .rewrapped_symmetric_key_associated_data()
                           .reencryption_public_key()));

    if (metadata_cwt.expiration_time.has_value()) {
      metadata_time = *metadata_cwt.expiration_time;
    }
  }
  if (other.has_hpke_plus_aead_data()) {
    FCP_ASSIGN_OR_RETURN(
        OkpCwt other_cwt,
        OkpCwt::Decode(other.hpke_plus_aead_data()
                           .rewrapped_symmetric_key_associated_data()
                           .reencryption_public_key()));
    if (other_cwt.expiration_time.has_value()) {
      other_time = *other_cwt.expiration_time;
    }
  }
  if (metadata_time < other_time) {
    return metadata;
  }
  return other;
}

BlobMetadata GetBlobMetadataFromRecord(const Record& record) {
  BlobMetadata metadata;
  metadata.set_compression_type(
      static_cast<BlobMetadata::CompressionType>(record.compression_type()));
  if (record.has_unencrypted_data()) {
    metadata.set_total_size_bytes(record.unencrypted_data().size());
    metadata.mutable_unencrypted();
    return metadata;
  }

  metadata.set_total_size_bytes(
      record.hpke_plus_aead_data().ciphertext().size());
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  const Record::HpkePlusAeadData& record_encryption_data =
      record.hpke_plus_aead_data();
  encryption_metadata->set_ciphertext_associated_data(
      record_encryption_data.ciphertext_associated_data());
  encryption_metadata->set_encrypted_symmetric_key(
      record_encryption_data.encrypted_symmetric_key());
  encryption_metadata->set_encapsulated_public_key(
      record_encryption_data.encapsulated_public_key());
  if (record_encryption_data.has_ledger_symmetric_key_associated_data()) {
    encryption_metadata->mutable_ledger_symmetric_key_associated_data()
        ->set_record_header(
            record_encryption_data.ledger_symmetric_key_associated_data()
                .record_header());
  } else {
    encryption_metadata->mutable_rewrapped_symmetric_key_associated_data()
        ->set_reencryption_public_key(
            record_encryption_data.rewrapped_symmetric_key_associated_data()
                .reencryption_public_key());

    encryption_metadata->mutable_rewrapped_symmetric_key_associated_data()
        ->set_nonce(
            record_encryption_data.rewrapped_symmetric_key_associated_data()
                .nonce());
  }
  return metadata;
}

}  // namespace confidential_federated_compute
