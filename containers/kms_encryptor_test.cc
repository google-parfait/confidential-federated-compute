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

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::NiceMock;

TEST(KmsEncryptorTest, EncryptIntermediateResult) {
  auto key_pair_1 = crypto_test_utils::GenerateKeyPair("key_1");
  auto key_pair_2 = crypto_test_utils::GenerateKeyPair("key_2");
  KmsEncryptor encryptor(
      std::vector<std::string>{key_pair_1.first, key_pair_2.first},
      "reencryption_policy_hash");

  BlobDecryptor blob_decryptor({key_pair_1.second, key_pair_2.second});

  // Encrypt using the first key
  absl::StatusOr<KmsEncryptor::EncryptedResult> encrypted_result =
      encryptor.EncryptIntermediateResult(0, "plaintext", "foo");
  ASSERT_THAT(encrypted_result, IsOk());

  // Decrypt using the same key
  EXPECT_THAT(blob_decryptor.DecryptBlob(encrypted_result->metadata,
                                         encrypted_result->ciphertext, "key_1"),
              IsOkAndHolds("plaintext"));
}

TEST(KmsEncryptorTest, EncryptIntermediateResultInvalidReencryptionIndex) {
  auto key_pair_1 = crypto_test_utils::GenerateKeyPair("key_pair_1");
  auto key_pair_2 = crypto_test_utils::GenerateKeyPair("key_pair_2");
  KmsEncryptor encryptor(
      std::vector<std::string>{key_pair_1.first, key_pair_2.first},
      "reencryption_policy_hash");
  EXPECT_THAT(encryptor.EncryptIntermediateResult(3, "plaintext", "foo"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(encryptor.EncryptIntermediateResult(-1, "plaintext", "bar"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace confidential_federated_compute
