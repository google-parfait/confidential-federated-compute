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
#include "containers/crypto.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/timestamp.pb.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::crypto_internal::UnwrapSymmetricKey;
using ::fcp::confidential_compute::crypto_internal::WrapSymmetricKey;
using ::fcp::confidential_compute::crypto_internal::WrapSymmetricKeyResult;
using ::fcp::confidentialcompute::Record;
using ::testing::HasSubstr;

absl::StatusOr<Record> CreateRewrappedRecord(
    absl::string_view message, absl::string_view ciphertext_associated_data,
    absl::string_view recipient_public_key, absl::string_view nonce,
    absl::string_view reencryption_public_key) {
  MessageEncryptor encryptor;
  // Encrypt the symmetric key with the public key of an intermediary.
  bssl::ScopedEVP_HPKE_KEY intermediary_key;
  const EVP_HPKE_KEM* kem = EVP_hpke_x25519_hkdf_sha256();
  const EVP_HPKE_KDF* kdf = EVP_hpke_hkdf_sha256();
  const EVP_HPKE_AEAD* aead = EVP_hpke_aes_128_gcm();
  FCP_CHECK(EVP_HPKE_KEY_generate(intermediary_key.get(), kem) == 1);
  size_t public_key_len;
  std::string intermediary_public_key(EVP_HPKE_MAX_PUBLIC_KEY_LENGTH, '\0');
  FCP_CHECK(EVP_HPKE_KEY_public_key(
                intermediary_key.get(),
                reinterpret_cast<uint8_t*>(intermediary_public_key.data()),
                &public_key_len, intermediary_public_key.size()) == 1);
  intermediary_public_key.resize(public_key_len);
  FCP_ASSIGN_OR_RETURN(EncryptMessageResult encrypt_result,
                       encryptor.Encrypt(message, intermediary_public_key,
                                         ciphertext_associated_data));

  // Have the intermediary rewrap the symmetric key with the public key of the
  // final recipient.
  FCP_ASSIGN_OR_RETURN(
      std::string symmetric_key,
      UnwrapSymmetricKey(intermediary_key.get(), kdf, aead,
                         encrypt_result.encrypted_symmetric_key,
                         encrypt_result.encapped_key,
                         ciphertext_associated_data));

  FCP_ASSIGN_OR_RETURN(
      WrapSymmetricKeyResult rewrapped_symmetric_key_result,
      WrapSymmetricKey(kem, kdf, aead, symmetric_key, recipient_public_key,
                       absl::StrCat(reencryption_public_key, nonce)));

  Record record;
  record.mutable_hpke_plus_aead_data()->set_ciphertext(
      encrypt_result.ciphertext);
  record.mutable_hpke_plus_aead_data()->set_ciphertext_associated_data(
      std::string(ciphertext_associated_data));
  record.mutable_hpke_plus_aead_data()->set_encrypted_symmetric_key(
      rewrapped_symmetric_key_result.encrypted_symmetric_key);
  record.mutable_hpke_plus_aead_data()->set_encapsulated_public_key(
      rewrapped_symmetric_key_result.encapped_key);
  record.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_nonce(std::string(nonce));
  record.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(std::string(reencryption_public_key));
  return record;
}

TEST(CryptoTest, EncryptAndDecrypt) {
  std::string message = "some plaintext message";
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  google::protobuf::Any config;
  RecordDecryptor record_decryptor(config);

  absl::StatusOr<const PublicKeyAndSignature*> public_key_and_signature =
      record_decryptor.GetPublicKeyAndSignature();
  ASSERT_TRUE(public_key_and_signature.ok())
      << public_key_and_signature.status();

  const std::string& recipient_public_key =
      public_key_and_signature.value()->public_key;

  absl::StatusOr<std::string> nonce = record_decryptor.GenerateNonce();
  ASSERT_TRUE(nonce.ok()) << nonce.status();

  absl::StatusOr<Record> rewrapped_record = CreateRewrappedRecord(
      message, ciphertext_associated_data, recipient_public_key, *nonce,
      reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();

  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(*rewrapped_record);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, EncryptAndDecryptWrongRecipient) {
  std::string message = "some plaintext message";
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  google::protobuf::Any config;
  RecordDecryptor record_decryptor(config);
  RecordDecryptor record_decryptor_other(config);

  // Use the public key from record_decryptor_other to rewrap the message.
  // record_decryptor will not be able to decrypt the record.
  absl::StatusOr<const PublicKeyAndSignature*> public_key_and_signature =
      record_decryptor_other.GetPublicKeyAndSignature();
  ASSERT_TRUE(public_key_and_signature.ok())
      << public_key_and_signature.status();

  const std::string& recipient_public_key =
      public_key_and_signature.value()->public_key;

  absl::StatusOr<std::string> nonce = record_decryptor.GenerateNonce();
  ASSERT_TRUE(nonce.ok()) << nonce.status();

  absl::StatusOr<Record> rewrapped_record = CreateRewrappedRecord(
      message, ciphertext_associated_data, recipient_public_key, *nonce,
      reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();

  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(*rewrapped_record);
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()));

  // record_decryptor_other should not be able to decrypt the record either,
  // since the nonce was generated by record_decryptor.
  absl::StatusOr<std::string> decrypt_result_other =
      record_decryptor_other.DecryptRecord(*rewrapped_record);
  ASSERT_FALSE(decrypt_result_other.ok());
  ASSERT_TRUE(absl::IsFailedPrecondition(decrypt_result_other.status()))
      << decrypt_result.status();
}

TEST(CryptoTest, DecryptUnencryptedData) {
  std::string message = "some plaintext message";
  google::protobuf::Any config;
  RecordDecryptor record_decryptor(config);

  Record record;
  record.set_unencrypted_data(message);
  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(record);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, DecryptRecordWithInvalidKind) {
  google::protobuf::Any config;
  RecordDecryptor record_decryptor(config);
  Record record;
  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(record);
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()))
      << decrypt_result.status();
}

TEST(CryptoTest, DecryptRecordWithInvalidAssociatedData) {
  google::protobuf::Any config;
  RecordDecryptor record_decryptor(config);
  Record record;
  record.mutable_hpke_plus_aead_data()->set_ciphertext("unused ciphertext");
  record.mutable_hpke_plus_aead_data()->set_encrypted_symmetric_key(
      "unused symmetric key");
  record.mutable_hpke_plus_aead_data()->set_encapsulated_public_key(
      "unused encapped key");
  // Setting the wrong kind of associated data will cause decryption to fail
  // early.
  record.mutable_hpke_plus_aead_data()
      ->mutable_ledger_symmetric_key_associated_data();
  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(record);
  ASSERT_FALSE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()))
      << decrypt_result.status();
  ASSERT_THAT(decrypt_result.status().message(),
              HasSubstr("Record to decrypt must contain "
                        "rewrapped_symmetric_key_associated_data"));
}

TEST(CryptoTest, DecryptTwiceFails) {
  std::string message = "some plaintext message";
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext_associated_data";

  google::protobuf::Any config;
  RecordDecryptor record_decryptor(config);

  absl::StatusOr<const PublicKeyAndSignature*> public_key_and_signature =
      record_decryptor.GetPublicKeyAndSignature();
  ASSERT_TRUE(public_key_and_signature.ok())
      << public_key_and_signature.status();

  const std::string& recipient_public_key =
      public_key_and_signature.value()->public_key;

  absl::StatusOr<std::string> nonce = record_decryptor.GenerateNonce();
  ASSERT_TRUE(nonce.ok()) << nonce.status();

  absl::StatusOr<Record> rewrapped_record = CreateRewrappedRecord(
      message, ciphertext_associated_data, recipient_public_key, *nonce,
      reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();

  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(*rewrapped_record);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);

  // Decrypting again will not succeed because the nonce should have been
  // deleted.
  absl::StatusOr<std::string> decrypt_result_2 =
      record_decryptor.DecryptRecord(*rewrapped_record);
  ASSERT_FALSE(decrypt_result_2.ok());
  ASSERT_TRUE(absl::IsFailedPrecondition(decrypt_result_2.status()))
      << decrypt_result_2.status();
}

TEST(CryptoTest, GetSignedPublicKeyAndConfigTwice) {
  google::protobuf::Any config;
  RecordDecryptor record_decryptor(config);

  absl::StatusOr<const PublicKeyAndSignature*> public_key_and_signature =
      record_decryptor.GetPublicKeyAndSignature();
  ASSERT_TRUE(public_key_and_signature.ok())
      << public_key_and_signature.status();

  // Now call GetPublicKeyAndSignature again.
  // This will succeed and return the same public key.
  absl::StatusOr<const PublicKeyAndSignature*> public_key_and_signature_2 =
      record_decryptor.GetPublicKeyAndSignature();
  ASSERT_TRUE(public_key_and_signature_2.ok())
      << public_key_and_signature_2.status();
  ASSERT_EQ((*public_key_and_signature)->public_key,
            (*public_key_and_signature_2)->public_key);
}

}  // namespace
}  // namespace confidential_federated_compute
