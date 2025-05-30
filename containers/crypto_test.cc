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

#include "absl/log/log.h"
#include "containers/blob_metadata.h"
#include "containers/crypto_test_utils.h"
#include "containers/session.h"
#include "fcp/base/compression.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/struct.pb.h"
#include "google/protobuf/util/message_differencer.h"
#include "grpcpp/support/status.h"
#include "gtest/gtest.h"
#include "proto/containers/orchestrator_crypto.pb.h"
#include "proto/containers/orchestrator_crypto_mock.grpc.pb.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::Record;
using ::oak::containers::v1::MockOrchestratorCryptoStub;
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

TEST(CryptoTest, EncryptAndDecrypt) {
  std::string message = "some plaintext message";
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);

  absl::StatusOr<absl::string_view> recipient_public_key =
      record_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok()) << recipient_public_key.status();

  absl::StatusOr<std::string> nonce = record_decryptor.GenerateNonce();
  ASSERT_TRUE(nonce.ok()) << nonce.status();

  absl::StatusOr<Record> rewrapped_record =
      crypto_test_utils::CreateRewrappedRecord(
          message, ciphertext_associated_data, *recipient_public_key, *nonce,
          reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();

  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(*rewrapped_record);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, EncryptAndDecryptWithGzipCompression) {
  std::string message = "some plaintext message";
  absl::StatusOr<std::string> compressed = fcp::CompressWithGzip(message);
  ASSERT_TRUE(compressed.ok()) << compressed.status();
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);

  absl::StatusOr<absl::string_view> recipient_public_key =
      record_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok()) << recipient_public_key.status();

  absl::StatusOr<std::string> nonce = record_decryptor.GenerateNonce();
  ASSERT_TRUE(nonce.ok()) << nonce.status();

  absl::StatusOr<Record> rewrapped_record =
      crypto_test_utils::CreateRewrappedRecord(
          *compressed, ciphertext_associated_data, *recipient_public_key,
          *nonce, reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();
  rewrapped_record->set_compression_type(Record::COMPRESSION_TYPE_GZIP);

  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(*rewrapped_record);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, EncryptAndDecryptWrongRecipient) {
  std::string message = "some plaintext message";
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);
  RecordDecryptor record_decryptor_other(mock_crypto_stub);

  // Use the public key from record_decryptor_other to rewrap the message.
  // record_decryptor will not be able to decrypt the record.
  absl::StatusOr<absl::string_view> recipient_public_key =
      record_decryptor_other.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok()) << recipient_public_key.status();

  absl::StatusOr<std::string> nonce = record_decryptor.GenerateNonce();
  ASSERT_TRUE(nonce.ok()) << nonce.status();

  absl::StatusOr<Record> rewrapped_record =
      crypto_test_utils::CreateRewrappedRecord(
          message, ciphertext_associated_data, *recipient_public_key, *nonce,
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
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);

  Record record;
  record.set_unencrypted_data(message);
  record.set_compression_type(Record::COMPRESSION_TYPE_NONE);
  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(record);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, DecryptUnencryptedDataWithGzipCompression) {
  std::string message = "some plaintext message";
  absl::StatusOr<std::string> compressed = fcp::CompressWithGzip(message);
  ASSERT_TRUE(compressed.ok()) << compressed.status();
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);

  Record record;
  record.set_unencrypted_data(*compressed);
  record.set_compression_type(Record::COMPRESSION_TYPE_GZIP);
  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(record);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, DecryptRecordWithInvalidKind) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);
  Record record;
  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(record);
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()))
      << decrypt_result.status();
}

TEST(CryptoTest, DecryptRecordWithoutCompressionType) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);

  Record record;
  record.set_unencrypted_data("some message");
  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(record);
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()))
      << decrypt_result.status();
}

TEST(CryptoTest, DecryptRecordWithInvalidGzipCompression) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);

  Record record;
  record.set_unencrypted_data("invalid");
  record.set_compression_type(Record::COMPRESSION_TYPE_GZIP);
  absl::StatusOr<std::string> decrypt_result =
      record_decryptor.DecryptRecord(record);
  ASSERT_FALSE(decrypt_result.ok());
}

TEST(CryptoTest, DecryptRecordWithInvalidAssociatedData) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);
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

  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);

  absl::StatusOr<absl::string_view> recipient_public_key =
      record_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok()) << recipient_public_key.status();

  absl::StatusOr<std::string> nonce = record_decryptor.GenerateNonce();
  ASSERT_TRUE(nonce.ok()) << nonce.status();

  absl::StatusOr<Record> rewrapped_record =
      crypto_test_utils::CreateRewrappedRecord(
          message, ciphertext_associated_data, *recipient_public_key, *nonce,
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

TEST(CryptoTest, GetPublicKey) {
  MockOrchestratorCryptoStub mock_crypto_stub;
  oak::containers::v1::SignRequest sign_request;
  oak::containers::v1::SignResponse sign_response;
  sign_response.mutable_signature()->set_signature("signature");
  EXPECT_CALL(mock_crypto_stub, Sign(_, _, _))
      .WillOnce(DoAll(SaveArg<1>(&sign_request),
                      SetArgPointee<2>(sign_response),
                      Return(grpc::Status::OK)));

  google::protobuf::Struct config_properties;
  (*config_properties.mutable_fields())["test"].set_bool_value(true);
  RecordDecryptor record_decryptor(mock_crypto_stub, config_properties);

  absl::StatusOr<absl::string_view> public_key =
      record_decryptor.GetPublicKey();
  ASSERT_TRUE(public_key.ok()) << public_key.status();
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(*public_key);
  ASSERT_TRUE(cwt.ok()) << cwt.status();
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(
      cwt->config_properties, config_properties))
      << "Actual: " << cwt->config_properties.DebugString();

  absl::StatusOr<std::string> sig_structure =
      cwt->BuildSigStructureForSigning(/*aad=*/"");
  ASSERT_TRUE(sig_structure.ok()) << sig_structure.status();
  EXPECT_EQ(sign_request.message(), *sig_structure);
  EXPECT_EQ(cwt->signature, "signature");
}

TEST(CryptoTest, GetPublicKeySigningFails) {
  MockOrchestratorCryptoStub mock_crypto_stub;
  EXPECT_CALL(mock_crypto_stub, Sign(_, _, _))
      .WillOnce(Return(grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "")));
  RecordDecryptor record_decryptor(mock_crypto_stub);

  EXPECT_EQ(record_decryptor.GetPublicKey().status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CryptoTest, GetPublicKeyTwice) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  RecordDecryptor record_decryptor(mock_crypto_stub);

  absl::StatusOr<absl::string_view> public_key =
      record_decryptor.GetPublicKey();
  ASSERT_TRUE(public_key.ok()) << public_key.status();

  // Now call GetPublicKey again.
  // This will succeed and return the same public key.
  absl::StatusOr<absl::string_view> public_key_2 =
      record_decryptor.GetPublicKey();
  ASSERT_TRUE(public_key_2.ok()) << public_key_2.status();
  ASSERT_EQ(*public_key, *public_key_2);
}

TEST(CryptoTest, EncryptAndDecryptBlob) {
  std::string message = "some plaintext message";
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);

  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok()) << recipient_public_key.status();

  absl::StatusOr<Record> rewrapped_record =
      crypto_test_utils::CreateRewrappedRecord(
          message, ciphertext_associated_data, *recipient_public_key, "nonce",
          reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();

  absl::StatusOr<std::string> decrypt_result = blob_decryptor.DecryptBlob(
      GetBlobMetadataFromRecord(*rewrapped_record),
      rewrapped_record->hpke_plus_aead_data().ciphertext());
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

  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  absl::flat_hash_set<std::string> authorized_policy_hashes;
  authorized_policy_hashes.insert("sha256_hash");
  BlobDecryptor blob_decryptor(mock_crypto_stub, {}, {private_key},
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

  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  absl::flat_hash_set<std::string> authorized_policy_hashes;
  authorized_policy_hashes.insert("sha256_hash");
  BlobDecryptor blob_decryptor(mock_crypto_stub, {}, {private_key},
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

  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);

  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok()) << recipient_public_key.status();

  absl::StatusOr<Record> rewrapped_record =
      crypto_test_utils::CreateRewrappedRecord(
          *compressed, ciphertext_associated_data, *recipient_public_key,
          "nonce", reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();
  rewrapped_record->set_compression_type(Record::COMPRESSION_TYPE_GZIP);

  absl::StatusOr<std::string> decrypt_result = blob_decryptor.DecryptBlob(
      GetBlobMetadataFromRecord(*rewrapped_record),
      rewrapped_record->hpke_plus_aead_data().ciphertext());
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, EncryptAndDecryptBlobWrongRecipient) {
  std::string message = "some plaintext message";
  std::string reencryption_public_key = "reencryption_public_key";
  std::string ciphertext_associated_data = "ciphertext associated data";

  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);
  BlobDecryptor blob_decryptor_other(mock_crypto_stub);

  // Use the public key from record_decryptor_other to rewrap the message.
  // record_decryptor will not be able to decrypt the record.
  absl::StatusOr<absl::string_view> recipient_public_key =
      blob_decryptor_other.GetPublicKey();
  ASSERT_TRUE(recipient_public_key.ok()) << recipient_public_key.status();

  absl::StatusOr<Record> rewrapped_record =
      crypto_test_utils::CreateRewrappedRecord(
          message, ciphertext_associated_data, *recipient_public_key, "nonce",
          reencryption_public_key);
  ASSERT_TRUE(rewrapped_record.ok()) << rewrapped_record.status();

  absl::StatusOr<std::string> decrypt_result = blob_decryptor.DecryptBlob(
      GetBlobMetadataFromRecord(*rewrapped_record),
      rewrapped_record->hpke_plus_aead_data().ciphertext());
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()));
}

TEST(CryptoTest, DecryptUnencryptedBlob) {
  std::string message = "some plaintext message";
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);

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
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);
  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, *compressed);
  ASSERT_TRUE(decrypt_result.ok()) << decrypt_result.status();
  ASSERT_EQ(*decrypt_result, message);
}

TEST(CryptoTest, DecryptBlobWithInvalidKind) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);
  BlobMetadata metadata;
  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, "message");
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()))
      << decrypt_result.status();
}

TEST(CryptoTest, DecryptBlobWithoutCompressionType) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, "message");
  ASSERT_FALSE(decrypt_result.ok());
  ASSERT_TRUE(absl::IsInvalidArgument(decrypt_result.status()))
      << decrypt_result.status();
}

TEST(CryptoTest, DecryptBlobWithInvalidGzipCompression) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);

  BlobMetadata metadata;
  metadata.mutable_unencrypted();
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);
  absl::StatusOr<std::string> decrypt_result =
      blob_decryptor.DecryptBlob(metadata, "message");
  ASSERT_FALSE(decrypt_result.ok());
}

TEST(CryptoTest, DecryptBlobWithInvalidAssociatedData) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);
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
  MockOrchestratorCryptoStub mock_crypto_stub;
  oak::containers::v1::SignRequest sign_request;
  oak::containers::v1::SignResponse sign_response;
  sign_response.mutable_signature()->set_signature("signature");
  EXPECT_CALL(mock_crypto_stub, Sign(_, _, _))
      .WillOnce(DoAll(SaveArg<1>(&sign_request),
                      SetArgPointee<2>(sign_response),
                      Return(grpc::Status::OK)));

  google::protobuf::Struct config_properties;
  (*config_properties.mutable_fields())["test"].set_bool_value(true);
  BlobDecryptor blob_decryptor(mock_crypto_stub, config_properties);

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
  EXPECT_EQ(sign_request.message(), *sig_structure);
  EXPECT_EQ(cwt->signature, "signature");
}

TEST(CryptoTest, BlobDecryptorGetPublicKeySigningFails) {
  MockOrchestratorCryptoStub mock_crypto_stub;
  EXPECT_CALL(mock_crypto_stub, Sign(_, _, _))
      .WillOnce(Return(grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "")));
  BlobDecryptor blob_decryptor(mock_crypto_stub);

  EXPECT_EQ(blob_decryptor.GetPublicKey().status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CryptoTest, BlobDecryptorGetPublicKeyTwice) {
  NiceMock<MockOrchestratorCryptoStub> mock_crypto_stub;
  BlobDecryptor blob_decryptor(mock_crypto_stub);

  absl::StatusOr<absl::string_view> public_key = blob_decryptor.GetPublicKey();
  ASSERT_TRUE(public_key.ok()) << public_key.status();

  // Now call GetPublicKey again.
  // This will succeed and return the same public key.
  absl::StatusOr<absl::string_view> public_key_2 =
      blob_decryptor.GetPublicKey();
  ASSERT_TRUE(public_key_2.ok()) << public_key_2.status();
  ASSERT_EQ(*public_key, *public_key_2);
}

TEST(CryptoTest, EncryptRecordSuccess) {
  MessageDecryptor decryptor;
  absl::StatusOr<std::string> recipient_public_key =
      decryptor.GetPublicKey([](absl::string_view) { return ""; }, 0);
  ASSERT_TRUE(recipient_public_key.ok());
  absl::StatusOr<OkpCwt> cwt = OkpCwt::Decode(*recipient_public_key);
  ASSERT_TRUE(cwt.ok());
  cwt->public_key->key_id = "key_id";

  absl::StatusOr<std::string> public_key_with_key_id = cwt->Encode();
  ASSERT_TRUE(public_key_with_key_id.ok());

  RecordEncryptor encryptor;
  absl::StatusOr<Record> encrypt_result =
      encryptor.EncryptRecord("plaintext", *public_key_with_key_id,
                              "policy_sha256", /*access_policy_node_id=*/1);

  ASSERT_TRUE(encrypt_result.ok());
  ASSERT_TRUE(encrypt_result->has_hpke_plus_aead_data());
  BlobHeader updated_header;
  updated_header.ParseFromString(
      encrypt_result->hpke_plus_aead_data().ciphertext_associated_data());
  ASSERT_EQ(updated_header.access_policy_node_id(), 1);
  ASSERT_NE(updated_header.blob_id(), "");
  ASSERT_EQ(updated_header.access_policy_sha256(), "policy_sha256");
  ASSERT_EQ(updated_header.key_id(), "key_id");

  absl::StatusOr<std::string> decrypted_result = decryptor.Decrypt(
      encrypt_result->hpke_plus_aead_data().ciphertext(),
      encrypt_result->hpke_plus_aead_data().ciphertext_associated_data(),
      encrypt_result->hpke_plus_aead_data().encrypted_symmetric_key(),
      encrypt_result->hpke_plus_aead_data().ciphertext_associated_data(),
      encrypt_result->hpke_plus_aead_data().encapsulated_public_key());
  ASSERT_TRUE(decrypted_result.ok()) << decrypted_result.status();
  ASSERT_EQ(*decrypted_result, "plaintext");
}

}  // namespace
}  // namespace confidential_federated_compute
