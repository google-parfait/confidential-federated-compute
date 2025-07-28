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

#include "absl/status/status_matchers.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

using ::absl_testing::IsOk;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;

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

  ASSERT_THAT(EarliestExpirationTimeMetadata(metadata, other), IsOk());
}

TEST(EarliestExpirationTimeMetadataTest,
     UnencryptedAndEncryptedWithoutExpirationTime) {
  BlobMetadata unencrypted;
  unencrypted.mutable_unencrypted();

  BlobMetadata encrypted_without_expiration;
  MessageDecryptor decryptor;
  absl::StatusOr<std::string> reencryption_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_THAT(reencryption_public_key, IsOk());
  encrypted_without_expiration.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_public_key);

  ASSERT_THAT(
      EarliestExpirationTimeMetadata(unencrypted, encrypted_without_expiration),
      IsOk());
}

TEST(EarliestExpirationTimeMetadataTest,
     UnencryptedAndEncryptedWithExpirationTime) {
  BlobMetadata unencrypted;
  unencrypted.mutable_unencrypted();

  BlobMetadata encrypted_with_expiration;
  absl::StatusOr<std::string> reencryption_key_with_expiration =
      GetKeyWithExpiration(absl::FromUnixSeconds(1));
  ASSERT_THAT(reencryption_key_with_expiration, IsOk());
  encrypted_with_expiration.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_key_with_expiration);

  absl::StatusOr<BlobMetadata> earliest =
      EarliestExpirationTimeMetadata(unencrypted, encrypted_with_expiration);
  ASSERT_THAT(earliest, IsOk());
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
  ASSERT_THAT(reencryption_public_key, IsOk());
  encrypted_without_expiration.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_public_key);

  BlobMetadata encrypted_with_expiration;
  absl::StatusOr<std::string> reencryption_key_with_expiration =
      GetKeyWithExpiration(absl::FromUnixSeconds(1));
  ASSERT_THAT(reencryption_key_with_expiration, IsOk());
  encrypted_with_expiration.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_key_with_expiration);

  absl::StatusOr<BlobMetadata> earliest = EarliestExpirationTimeMetadata(
      encrypted_without_expiration, encrypted_with_expiration);
  ASSERT_THAT(earliest, IsOk());
  ASSERT_EQ(earliest->hpke_plus_aead_data()
                .rewrapped_symmetric_key_associated_data()
                .reencryption_public_key(),
            *reencryption_key_with_expiration);
}

TEST(EarliestExpirationTimeMetadataTest, BothEncryptedWithExpirationTime) {
  BlobMetadata earlier_metadata;
  absl::StatusOr<std::string> earlier_key =
      GetKeyWithExpiration(absl::FromUnixSeconds(1));
  ASSERT_THAT(earlier_key, IsOk());
  earlier_metadata.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*earlier_key);

  BlobMetadata later_metadata;
  absl::StatusOr<std::string> later_key =
      GetKeyWithExpiration(absl::FromUnixSeconds(42));
  ASSERT_THAT(later_key, IsOk());
  later_metadata.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*later_key);

  absl::StatusOr<BlobMetadata> earliest =
      EarliestExpirationTimeMetadata(later_metadata, earlier_metadata);
  ASSERT_THAT(earliest, IsOk());
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
  ASSERT_THAT(reencryption_public_key_1, IsOk());
  encrypted_without_expiration_1.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_public_key_1);

  BlobMetadata encrypted_without_expiration_2;
  MessageDecryptor decryptor_2;
  absl::StatusOr<std::string> reencryption_public_key_2 =
      decryptor_2.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_THAT(reencryption_public_key_2, IsOk());
  encrypted_without_expiration_2.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(*reencryption_public_key_2);

  ASSERT_THAT(EarliestExpirationTimeMetadata(encrypted_without_expiration_1,
                                             encrypted_without_expiration_2),
              IsOk());
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

TEST(GetKeyIdFromMetadata, SuccessUnencrypted) {
  BlobMetadata unencrypted;
  unencrypted.mutable_unencrypted();
  EXPECT_EQ(GetKeyIdFromMetadata(unencrypted).value(), "");
}

TEST(GetKeyIdFromMetadata, SuccessEncrypted) {
  BlobHeader header;
  header.set_blob_id("blob_id");
  header.set_key_id("key_id");
  BlobMetadata metadata;
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header(header.SerializeAsString());

  EXPECT_EQ(GetKeyIdFromMetadata(metadata).value(), "key_id");
}

TEST(GetKeyIdFromMetadata, InvalidAssociatedData) {
  BlobMetadata metadata;
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header("invalid!!!");

  EXPECT_EQ(GetKeyIdFromMetadata(metadata).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(GetKeyIdFromMetadata, NoKmsAssociatedData) {
  BlobMetadata metadata;
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key("some_key");

  EXPECT_EQ(GetKeyIdFromMetadata(metadata).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace confidential_federated_compute
