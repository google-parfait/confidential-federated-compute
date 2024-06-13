// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "containers/blob_metadata.h"

#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::Record;

TEST(GetBlobMetadataFromRecordTest, UnencryptedRecord) {
  Record unencrypted_record;
  unencrypted_record.set_compression_type(Record::COMPRESSION_TYPE_GZIP);
  unencrypted_record.set_unencrypted_data("data");
  BlobMetadata metadata = GetBlobMetadataFromRecord(unencrypted_record);
  ASSERT_TRUE(metadata.has_unencrypted());
  ASSERT_EQ(metadata.compression_type(), BlobMetadata::COMPRESSION_TYPE_GZIP);
}
TEST(GetBlobMetadataFromRecordTest, EncryptedRecord) {
  Record::HpkePlusAeadData hpke_plus_aead_data;
  hpke_plus_aead_data.set_ciphertext("ciphertext");
  hpke_plus_aead_data.set_ciphertext_associated_data("associated data");
  hpke_plus_aead_data.set_encrypted_symmetric_key("symmetric key");
  hpke_plus_aead_data.set_encapsulated_public_key("encapsulated key");
  hpke_plus_aead_data.mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key("reencryption key");
  hpke_plus_aead_data.mutable_rewrapped_symmetric_key_associated_data()
      ->set_nonce("nonce");
  Record encrypted_record;
  *encrypted_record.mutable_hpke_plus_aead_data() = hpke_plus_aead_data;
  encrypted_record.set_compression_type(Record::COMPRESSION_TYPE_GZIP);

  BlobMetadata metadata = GetBlobMetadataFromRecord(encrypted_record);

  ASSERT_EQ(metadata.hpke_plus_aead_data().ciphertext_associated_data(),
            "associated data");
  ASSERT_EQ(metadata.hpke_plus_aead_data().encrypted_symmetric_key(),
            "symmetric key");
  ASSERT_EQ(metadata.hpke_plus_aead_data().encapsulated_public_key(),
            "encapsulated key");
  ASSERT_EQ(metadata.hpke_plus_aead_data()
                .rewrapped_symmetric_key_associated_data()
                .reencryption_public_key(),
            "reencryption key");
  ASSERT_EQ(metadata.hpke_plus_aead_data()
                .rewrapped_symmetric_key_associated_data()
                .nonce(),
            "nonce");
  ASSERT_EQ(metadata.compression_type(), BlobMetadata::COMPRESSION_TYPE_GZIP);
}

absl::StatusOr<std::string> GetKeyWithExpiration(const absl::Time& expiration) {
  MessageDecryptor decryptor;
  FCP_ASSIGN_OR_RETURN(
      std::string reencryption_public_key,
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0));
  FCP_ASSIGN_OR_RETURN(OkpCwt reencryption_okp_cwt,
                       OkpCwt::Decode(reencryption_public_key));
  reencryption_okp_cwt.expiration_time = expiration;
  return reencryption_okp_cwt.Encode();
}

TEST(EarliestExpirationTimeMetadataTest, BothUnencrypted) {
  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  BlobMetadata other;
  other.mutable_unencrypted();

  ASSERT_TRUE(EarliestExpirationTimeMetadata(metadata, other).ok());
}

TEST(EarliestExpirationTimeMetadataTest,
     UnencryptedAndEncryptedWithoutExpirationTime) {
  BlobMetadata unencrypted;
  unencrypted.mutable_unencrypted();

  BlobMetadata encrypted_without_expiration;
  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());
  encrypted_without_expiration.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_public_key);

  ASSERT_TRUE(
      EarliestExpirationTimeMetadata(unencrypted, encrypted_without_expiration)
          .ok());
}

TEST(EarliestExpirationTimeMetadataTest,
     UnencryptedAndEncryptedWithExpirationTime) {
  BlobMetadata unencrypted;
  unencrypted.mutable_unencrypted();

  BlobMetadata encrypted_with_expiration;
  absl::StatusOr<std::string> reencryption_key_with_expiration =
      GetKeyWithExpiration(absl::FromUnixSeconds(1));
  ASSERT_TRUE(reencryption_key_with_expiration.ok());
  encrypted_with_expiration.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_key_with_expiration);

  absl::StatusOr<BlobMetadata> earliest =
      EarliestExpirationTimeMetadata(unencrypted, encrypted_with_expiration);
  ASSERT_TRUE(earliest.ok());
  ASSERT_EQ(earliest->hpke_plus_aead_data()
                .rewrapped_symmetric_key_associated_data()
                .reencryption_public_key(),
            *reencryption_key_with_expiration);
}

TEST(EarliestExpirationTimeMetadataTest,
     EncryptedWithoutExpirationTimeAndEncryptedWith) {
  BlobMetadata encrypted_without_expiration;
  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key.ok());
  encrypted_without_expiration.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_public_key);

  BlobMetadata encrypted_with_expiration;
  absl::StatusOr<std::string> reencryption_key_with_expiration =
      GetKeyWithExpiration(absl::FromUnixSeconds(1));
  ASSERT_TRUE(reencryption_key_with_expiration.ok());
  encrypted_with_expiration.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_key_with_expiration);

  absl::StatusOr<BlobMetadata> earliest = EarliestExpirationTimeMetadata(
      encrypted_without_expiration, encrypted_with_expiration);
  ASSERT_TRUE(earliest.ok());
  ASSERT_EQ(earliest->hpke_plus_aead_data()
                .rewrapped_symmetric_key_associated_data()
                .reencryption_public_key(),
            *reencryption_key_with_expiration);
}

TEST(EarliestExpirationTimeMetadataTest, BothEncryptedWithExpirationTime) {
  BlobMetadata earlier_metadata;
  absl::StatusOr<std::string> earlier_key =
      GetKeyWithExpiration(absl::FromUnixSeconds(1));
  ASSERT_TRUE(earlier_key.ok());
  earlier_metadata.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*earlier_key);

  BlobMetadata later_metadata;
  absl::StatusOr<std::string> later_key =
      GetKeyWithExpiration(absl::FromUnixSeconds(42));
  ASSERT_TRUE(later_key.ok());
  later_metadata.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*later_key);

  absl::StatusOr<BlobMetadata> earliest =
      EarliestExpirationTimeMetadata(later_metadata, earlier_metadata);
  ASSERT_TRUE(earliest.ok());
  ASSERT_EQ(earliest->hpke_plus_aead_data()
                .rewrapped_symmetric_key_associated_data()
                .reencryption_public_key(),
            *earlier_key);
}
TEST(EarliestExpirationTimeMetadataTest, BothEncryptedWithoutExpirationTime) {
  BlobMetadata encrypted_without_expiration_1;
  MessageDecryptor decryptor_1;
  absl::StatusOr<std::string> reencryption_public_key_1 =
      decryptor_1.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key_1.ok());
  encrypted_without_expiration_1.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_public_key_1);

  BlobMetadata encrypted_without_expiration_2;
  MessageDecryptor decryptor_2;
  absl::StatusOr<std::string> reencryption_public_key_2 =
      decryptor_2.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(reencryption_public_key_2.ok());
  encrypted_without_expiration_2.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_public_key_2);

  ASSERT_TRUE(EarliestExpirationTimeMetadata(encrypted_without_expiration_1,
                                             encrypted_without_expiration_2)
                  .ok());
}

TEST(EarliestExpirationTimeMetadataTest, UndecodeableKeysFail) {
  BlobMetadata unencrypted;
  unencrypted.mutable_unencrypted();

  BlobMetadata metadata_with_bad_key;
  metadata_with_bad_key.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key("bad key");

  ASSERT_FALSE(
      EarliestExpirationTimeMetadata(unencrypted, metadata_with_bad_key).ok());
}

}  // namespace
}  // namespace confidential_federated_compute
