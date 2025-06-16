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

#include <string>
#include <tuple>

#include "absl/log/log.h"
#include "cc/crypto/signing_key.h"
#include "containers/crypto_test_utils.h"
#include "containers/session.h"
#include "fcp/base/compression.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/struct.pb.h"
#include "google/protobuf/util/message_differencer.h"
#include "grpcpp/support/status.h"
#include "gtest/gtest.h"
#include "proto/crypto/crypto.pb.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::testing::_;
using ::testing::DoAll;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::SetArgPointee;

TEST(CryptoTest, KeyedHashSuccess) {
  absl::string_view input = "The quick brown fox jumps over the lazy dog";
  absl::string_view key = "key";
  absl::StatusOr<std::string> result = KeyedHash(input, key);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->size(), 32);
  EXPECT_TRUE(*result != input);
}

TEST(CryptoTest, KeyedHashEmptyInput) {
  absl::string_view input = "";
  absl::string_view key = "key";
  absl::StatusOr<std::string> result = KeyedHash(input, key);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->size(), 32);
  EXPECT_TRUE(*result != input);
}

TEST(CryptoTest, KeyedHashEmptyKey) {
  absl::string_view input = "foo";
  absl::string_view key = "";
  absl::StatusOr<std::string> result = KeyedHash(input, key);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->size(), 32);
  EXPECT_TRUE(*result != input);
}

TEST(CryptoTest, KeyedHashBothEmpty) {
  absl::string_view input = "";
  absl::string_view key = "";
  absl::StatusOr<std::string> result = KeyedHash(input, key);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->size(), 32);
  EXPECT_TRUE(*result != input);
}

TEST(CryptoTest, EncryptAndDecryptBlob) {
  std::string message = "some plaintext message";
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok()) << recipient_public_key.status();

  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob =
      crypto_test_utils::CreateRewrappedBlob(
          message, ciphertext_associated_data, *recipient_public_key, "nonce",
          reencryption_public_key);
  ASSERT_TRUE(rewrapped_blob.ok()) << rewrapped_blob.status();

  absl::StatusOr<std::string> decrypt_result = blob_decryptor.DecryptBlob(
      std::get<0>(*rewrapped_blob), std::get<1>(*rewrapped_blob));
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, EncryptAndDecryptBlobWithKmsProvidedKeys) {
  std::string message = "some plaintext message";
  std::string key_id = "some key id";
  auto [public_key, private_key] = crypto_test_utils::GenerateKeyPair(key_id);
  BlobHeader header;
  header.set_key_id(key_id);
  header.set_access_policy_sha256("sha256_hash");
  std::string associated_data = header.SerializeAsString();

  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  absl::flat_hash_set<std::string> authorized_policy_hashes;
  authorized_policy_hashes.insert("sha256_hash");
  BlobDecryptor blob_decryptor(mock_signing_key_handle, {}, {private_key},
                               authorized_policy_hashes);

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, public_key, associated_data);
  ASSERT_TRUE(encrypt_result.ok()) << encrypt_result.status();

  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(encrypt_result.value().ciphertext.size());
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->set_ciphertext_associated_data(associated_data);
  encryption_metadata->set_encrypted_symmetric_key(
      encrypt_result.value().encrypted_symmetric_key);
  encryption_metadata->set_encapsulated_public_key(
      encrypt_result.value().encapped_key);
  encryption_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header(associated_data);

  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, encrypt_result.value().ciphertext);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, EncryptAndDecryptBlobWithKmsInvalidPolicyHash) {
  std::string message = "some plaintext message";
  std::string key_id = "some key id";
  auto [public_key, private_key] = crypto_test_utils::GenerateKeyPair(key_id);
  BlobHeader header;
  header.set_key_id(key_id);
  header.set_access_policy_sha256("invalid_hash");
  std::string associated_data = header.SerializeAsString();

  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  absl::flat_hash_set<std::string> authorized_policy_hashes;
  authorized_policy_hashes.insert("sha256_hash");
  BlobDecryptor blob_decryptor(mock_signing_key_handle, {}, {private_key},
                               authorized_policy_hashes);

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, public_key, associated_data);
  ASSERT_TRUE(encrypt_result.ok()) << encrypt_result.status();

  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(encrypt_result.value().ciphertext.size());
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->set_ciphertext_associated_data(associated_data);
  encryption_metadata->set_encrypted_symmetric_key(
      encrypt_result.value().encrypted_symmetric_key);
  encryption_metadata->set_encapsulated_public_key(
      encrypt_result.value().encapped_key);
  encryption_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header(associated_data);

  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, encrypt_result.value().ciphertext);
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()));
}

TEST(CryptoTest, EncryptAndDecryptBlobWithGzipCompression) {
  std::string message = "some plaintext message";
  absl::StatusOr<std::string> compressed = fcp::CompressWithGzip(message);
  ASSERT_TRUE(compressed.ok()) << compressed.status();
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok()) << recipient_public_key.status();

  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob =
      crypto_test_utils::CreateRewrappedBlob(
          *compressed, ciphertext_associated_data, *recipient_public_key,
          "nonce", reencryption_public_key);
  ASSERT_TRUE(rewrapped_blob.ok()) << rewrapped_blob.status();
  std::get<0>(*rewrapped_blob)
      .set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);

  absl::StatusOr<std::string> decrypt_result = blob_decryptor.DecryptBlob(
      std::get<0>(*rewrapped_blob), std::get<1>(*rewrapped_blob));
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, EncryptAndDecryptBlobWrongRecipient) {
  std::string message = "some plaintext message";
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  BlobDecryptor blob_decryptor_other(mock_signing_key_handle);

  // Use the public key from blob_decryptor_other to rewrap the message.
  // blob_decryptor will not be able to decrypt the blob.
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor_other.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok()) << recipient_public_key.status();

  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob =
      crypto_test_utils::CreateRewrappedBlob(
          message, ciphertext_associated_data, *recipient_public_key, "nonce",
          reencryption_public_key);
  ASSERT_TRUE(rewrapped_blob.ok()) << rewrapped_blob.status();

  absl::StatusOr<std::string> decrypt_result = blob_decryptor.DecryptBlob(
      std::get<0>(*rewrapped_blob), std::get<1>(*rewrapped_blob));
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()));
}

TEST(CryptoTest, DecryptUnencryptedBlob) {
  std::string message = "some plaintext message";
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, message);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, DecryptUnencryptedBlobWithGzipCompression) {
  std::string message = "some plaintext message";
  absl::StatusOr<std::string> compressed = fcp::CompressWithGzip(message);
  ASSERT_TRUE(compressed.ok()) << compressed.status();
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);
  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, *compressed);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, DecryptBlobWithInvalidKind) {
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  BlobMetadata metadata;
  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, "message");
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()))
      << decrypt_result.status();
}

TEST(CryptoTest, DecryptBlobWithoutCompressionType) {
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, "message");
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()))
      << decrypt_result.status();
}

TEST(CryptoTest, DecryptBlobWithInvalidGzipCompression) {
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);
  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, "message");
  ASSERT_FALSE(decrypt_result.ok());
}

TEST(CryptoTest, DecryptBlobWithInvalidAssociatedData) {
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  BlobMetadata metadata;
  metadata.mutable_hpke_plus_aead_data()->set_encrypted_symmetric_key(
      "unused symmetric key");
  metadata.mutable_hpke_plus_aead_data()->set_encapsulated_public_key(
      "unused encapped key");
  // Setting the wrong kind of associated data will cause decryption to fail
  // early.
  metadata.mutable_hpke_plus_aead_data()
      ->mutable_ledger_symmetric_key_associated_data();
  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, "unused message");
  ASSERT_FALSE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()))
      << decrypt_result.status();
  ASSERT_THAT(decrypt_result.status().message(),
              HasSubstr("Blob to decrypt must contain "
                        "either rewrapped_symmetric_key_associated_data or "
                        "kms_symmetric_key_associated_data"));
}

TEST(CryptoTest, BlobDecryptorGetPublicKey) {
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  std::string sign_request;
  oak::crypto::v1::Signature signature;
  signature.set_signature("signature");
  EXPECT_CALL(mock_signing_key_handle, Sign(_))
      .WillOnce(DoAll(SaveArg<0>(&sign_request), Return(signature)));

  google::protobuf::Struct config_properties;
  (*config_properties.mutable_fields())["test"].set_bool_value(true);
  BlobDecryptor blob_decryptor(mock_signing_key_handle, config_properties);

  absl::StatusOr<absl::string_view> public_key = blob_decryptor.GetPublicKey();
  ASSERT_TRUE(public_key.ok()) << public_key.status();
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(*public_key);
  ASSERT_TRUE(cwt.ok()) << cwt.status();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      cwt->config_properties, config_properties))
      << "Actual: " << cwt->config_properties.DebugString();

  absl::StatusOr<std::string> sig_structure =
      cwt->BuildSigStructureForSigning(/*aad=*/"");
  ASSERT_TRUE(sig_structure.ok()) << sig_structure.status();
  EXPECT_EQ(sign_request, *sig_structure);
  EXPECT_EQ(cwt->signature, "signature");
}

TEST(CryptoTest, BlobDecryptorGetPublicKeySigningFails) {
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  EXPECT_CALL(mock_signing_key_handle, Sign(_))
      .WillOnce(Return(absl::InvalidArgumentError("")));
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  EXPECT_EQ(blob_decryptor.GetPublicKey().status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CryptoTest, BlobDecryptorGetPublicKeyTwice) {
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  EXPECT_CALL(mock_signing_key_handle, Sign(_))
      .WillOnce(Return(oak::crypto::v1::Signature()));
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  absl::StatusOr<absl::string_view> public_key = blob_decryptor.GetPublicKey();
  ASSERT_TRUE(public_key.ok()) << public_key.status();

  // Now call GetPublicKey again.
  // This will succeed and return the same public key.
  absl::StatusOr<absl::string_view> public_key_2 =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(public_key_2.ok()) << public_key_2.status();
  ASSERT_EQ(*public_key, *public_key_2);
}

}  // namespace
}  // namespace confidential_federated_compute
