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
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::testing::_;
using ::testing::DoAll;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::NiceMock;
using ::testing::Not;
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
  std::string key_id = "some key id";
  auto [public_key, private_key] = crypto_test_utils::GenerateKeyPair(key_id);
  std::string associated_data = "associated_data";

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

  Decryptor decryptor({private_key});
  EXPECT_THAT(decryptor.DecryptBlob(metadata, encrypt_result.value().ciphertext,
                                    key_id),
              IsOkAndHolds(message));
}

TEST(CryptoTest, UnwrapReleaseToken) {
  auto [public_key, private_key] = crypto_test_utils::GenerateKeyPair("key-id");
  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> result = encryptor.EncryptForRelease(
      "result", public_key, "associated-data", "src", "dst",
      [](absl::string_view) { return "signature"; });
  ASSERT_THAT(result, IsOk());

  Decryptor decryptor({private_key});
  auto unwrapped_token = decryptor.UnwrapReleaseToken(result->release_token);
  EXPECT_THAT(unwrapped_token, IsOk());
  EXPECT_EQ(unwrapped_token->src_state, std::optional<std::string>("src"));
  EXPECT_EQ(unwrapped_token->dst_state, "dst");
  EXPECT_THAT(unwrapped_token->serialized_symmetric_key, Not(IsEmpty()));
}

TEST(CryptoTest, EncryptAndDecryptBlobWithGzipCompression) {
  std::string message = "some plaintext message";
  absl::StatusOr<std::string> compressed = fcp::CompressWithGzip(message);
  ASSERT_THAT(compressed, IsOk());
  std::string key_id = "some key id";
  auto [public_key, private_key] = crypto_test_utils::GenerateKeyPair(key_id);
  std::string associated_data = "associated_data";

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(*compressed, public_key, associated_data);
  ASSERT_THAT(encrypt_result, IsOk());

  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);
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

  Decryptor decryptor({private_key});
  EXPECT_THAT(decryptor.DecryptBlob(metadata, encrypt_result.value().ciphertext,
                                    key_id),
              IsOkAndHolds(message));
}

TEST(CryptoTest, EncryptAndDecryptBlobWrongKeyId) {
  std::string message = "some plaintext message";
  absl::StatusOr<std::string> compressed = fcp::CompressWithGzip(message);
  ASSERT_THAT(compressed, IsOk());
  std::string key_id = "some key id";
  auto [public_key, private_key] = crypto_test_utils::GenerateKeyPair(key_id);
  std::string associated_data = "associated_data";

  MessageEncryptor encryptor;
  absl::StatusOr<EncryptMessageResult> encrypt_result =
      encryptor.Encrypt(*compressed, public_key, associated_data);
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

  Decryptor decryptor({private_key});
  EXPECT_THAT(decryptor.DecryptBlob(metadata, encrypt_result.value().ciphertext,
                                    "invalid key id"),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(CryptoTest, DecryptUnencryptedBlob) {
  std::string message = "some plaintext message";
  Decryptor decryptor;

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  EXPECT_THAT(decryptor.DecryptBlob(metadata, message), IsOkAndHolds(message));
}

TEST(CryptoTest, DecryptUnencryptedBlobWithGzipCompression) {
  std::string message = "some plaintext message";
  absl::StatusOr<std::string> compressed = fcp::CompressWithGzip(message);
  ASSERT_THAT(compressed, IsOk());
  Decryptor decryptor;

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);
  EXPECT_THAT(decryptor.DecryptBlob(metadata, *compressed),
              IsOkAndHolds(message));
}

TEST(CryptoTest, DecryptBlobWithInvalidKind) {
  Decryptor decryptor;
  BlobMetadata metadata;
  EXPECT_THAT(decryptor.DecryptBlob(metadata, "message"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CryptoTest, DecryptBlobWithoutCompressionType) {
  Decryptor decryptor;

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  EXPECT_THAT(decryptor.DecryptBlob(metadata, "message"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CryptoTest, DecryptBlobWithInvalidGzipCompression) {
  Decryptor decryptor;
  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);
  EXPECT_FALSE(decryptor.DecryptBlob(metadata, "message").ok());
}

TEST(CryptoTest, DecryptBlobWithInvalidAssociatedData) {
  Decryptor decryptor;
  BlobMetadata metadata;
  metadata.mutable_hpke_plus_aead_data()->set_encrypted_symmetric_key(
      "unused symmetric key");
  metadata.mutable_hpke_plus_aead_data()->set_encapsulated_public_key(
      "unused encapped key");
  // No `kms_symmetric_key_associated_data` set.

  EXPECT_THAT(decryptor.DecryptBlob(metadata, "unused message"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Blob to decrypt must contain "
                                 "kms_symmetric_key_associated_data")));
}

}  // namespace
}  // namespace confidential_federated_compute
