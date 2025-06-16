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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CRYPTO_TEST_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CRYPTO_TEST_UTILS_H_

#include <string>
#include <tuple>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "cc/crypto/signing_key.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "proto/crypto/crypto.pb.h"

namespace confidential_federated_compute::crypto_test_utils {

// Helper function for tests that need to obtain a BlobMetadata + ciphertext
// which is in the format that would be produced if the Ledger carried out the
// encryption protocol in order to give a component having
// `recipient_public_key` access to the underlying plaintext message.
absl::StatusOr<std::tuple<fcp::confidentialcompute::BlobMetadata, std::string>>
CreateRewrappedBlob(absl::string_view message,
                    absl::string_view ciphertext_associated_data,
                    absl::string_view recipient_public_key,
                    absl::string_view nonce,
                    absl::string_view reencryption_public_key);

// Helper function to generate public and private key pairs which are in a
// format that would be produced by KMS for encrypting/decrypting blobs using
// the given key_id.
std::pair<std::string, std::string> GenerateKeyPair(std::string key_id);

// Mock SigningKeyHandle.
class MockSigningKeyHandle : public oak::crypto::SigningKeyHandle {
 public:
  MockSigningKeyHandle();

  MOCK_METHOD(absl::StatusOr<oak::crypto::v1::Signature>, Sign,
              (absl::string_view message), (override));
};

}  // namespace confidential_federated_compute::crypto_test_utils

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CRYPTO_TEST_UTILS_H_
