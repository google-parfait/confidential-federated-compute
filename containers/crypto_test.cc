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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
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

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
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

TEST(CryptoTest, RandomBlobId) {
  std::string blob_id1 = RandomBlobId();
  std::string blob_id2 = RandomBlobId();
  EXPECT_EQ(blob_id1.size(), 16);
  EXPECT_EQ(blob_id2.size(), 16);
  EXPECT_NE(blob_id1, blob_id2);
}

TEST(CryptoTest, KeyedHashSuccess) {
  absl::string_view input = "The quick brown fox jumps over the lazy dog";
  absl::string_view key = "key";
  absl::StatusOr<std::string> result = KeyedHash(input, key);
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->size(), 32);
  EXPECT_TRUE(*result != input);
}

TEST(CryptoTest, KeyedHashEmptyInput) {
  absl::string_view input = "";
  absl::string_view key = "key";
  absl::StatusOr<std::string> result = KeyedHash(input, key);
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->size(), 32);
  EXPECT_TRUE(*result != input);
}

TEST(CryptoTest, KeyedHashEmptyKey) {
  absl::string_view input = "foo";
  absl::string_view key = "";
  absl::StatusOr<std::string> result = KeyedHash(input, key);
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->size(), 32);
  EXPECT_TRUE(*result != input);
}

TEST(CryptoTest, KeyedHashBothEmpty) {
  absl::string_view input = "";
  absl::string_view key = "";
  absl::StatusOr<std::string> result = KeyedHash(input, key);
  ASSERT_THAT(result, IsOk());
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
  ASSERT_THAT(recipient_public_key, IsOk());

  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob =
      crypto_test_utils::CreateRewrappedBlob(
          message, ciphertext_associated_data, *recipient_public_key, "nonce",
          reencryption_public_key);
  ASSERT_THAT(rewrapped_blob, IsOk());

  EXPECT_THAT(blob_decryptor.DecryptBlob(std::get<0>(*rewrapped_blob),
                                         std::get<1>(*rewrapped_blob)),
              IsOkAndHolds(message));
}

TEST(CryptoTest, EncryptAndDecryptBlobWithKmsKeyId) {
  std::string message = "some plaintext message";
  std::string key_id = "some key id";
  auto [public_key, private_key] = crypto_test_utils::GenerateKeyPair(key_id);
  std::string associated_data = "associated_data";

  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle, {}, {private_key});

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(message, public_key, associated_data);
  ASSERT_THAT(encrypt_result, IsOk());

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

  EXPECT_THAT(blob_decryptor.DecryptBlob(
                  metadata, encrypt_result.value().ciphertext, key_id),
              IsOkAndHolds(message));
}

TEST(CryptoTest, EncryptAndDecryptBlobWithGzipCompression) {
  std::string message = "some plaintext message";
  absl::StatusOr<std::string> compressed = fcp::CompressWithGzip(message);
  ASSERT_THAT(compressed, IsOk());
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_THAT(recipient_public_key, IsOk());

  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob =
      crypto_test_utils::CreateRewrappedBlob(
          *compressed, ciphertext_associated_data, *recipient_public_key,
          "nonce", reencryption_public_key);
  ASSERT_THAT(rewrapped_blob, IsOk());
  std::get<0>(*rewrapped_blob)
      .set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);

  EXPECT_THAT(blob_decryptor.DecryptBlob(std::get<0>(*rewrapped_blob),
                                         std::get<1>(*rewrapped_blob)),
              IsOkAndHolds(message));
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
  ASSERT_THAT(recipient_public_key, IsOk());

  absl::StatusOr<std::tuple<BlobMetadata, std::string>> rewrapped_blob =
      crypto_test_utils::CreateRewrappedBlob(
          message, ciphertext_associated_data, *recipient_public_key, "nonce",
          reencryption_public_key);
  ASSERT_THAT(rewrapped_blob, IsOk());

  EXPECT_THAT(blob_decryptor.DecryptBlob(std::get<0>(*rewrapped_blob),
                                         std::get<1>(*rewrapped_blob)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CryptoTest, DecryptUnencryptedBlob) {
  std::string message = "some plaintext message";
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  EXPECT_THAT(blob_decryptor.DecryptBlob(metadata, message),
              IsOkAndHolds(message));
}

TEST(CryptoTest, DecryptUnencryptedBlobWithGzipCompression) {
  std::string message = "some plaintext message";
  absl::StatusOr<std::string> compressed = fcp::CompressWithGzip(message);
  ASSERT_THAT(compressed, IsOk());
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);
  EXPECT_THAT(blob_decryptor.DecryptBlob(metadata, *compressed),
              IsOkAndHolds(message));
}

TEST(CryptoTest, DecryptBlobWithInvalidKind) {
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  BlobMetadata metadata;
  EXPECT_THAT(blob_decryptor.DecryptBlob(metadata, "message"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CryptoTest, DecryptBlobWithoutCompressionType) {
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  EXPECT_THAT(blob_decryptor.DecryptBlob(metadata, "message"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CryptoTest, DecryptBlobWithInvalidGzipCompression) {
  NiceMock<crypto_test_utils::MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);
  EXPECT_FALSE(blob_decryptor.DecryptBlob(metadata, "message").ok());
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
  EXPECT_THAT(blob_decryptor.DecryptBlob(metadata, "unused message"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Blob to decrypt must contain either "
                                 "rewrapped_symmetric_key_associated_data or "
                                 "kms_symmetric_key_associated_data")));
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
  ASSERT_THAT(public_key, IsOk());
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(*public_key);
  ASSERT_THAT(cwt, IsOk());
  google::protobuf::Struct cwt_config_properties;
  ASSERT_TRUE(cwt_config_properties.ParseFromString(cwt->config_properties));
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      cwt_config_properties, config_properties))
      << "Actual: " << cwt_config_properties.DebugString();

  EXPECT_THAT(cwt->BuildSigStructureForSigning(/*aad=*/""),
              IsOkAndHolds(sign_request));
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
  ASSERT_THAT(public_key, IsOk());

  // Now call GetPublicKey again.
  // This will succeed and return the same public key.
  EXPECT_THAT(blob_decryptor.GetPublicKey(), IsOkAndHolds(*public_key));
}

}  // namespace
}  // namespace confidential_federated_compute
