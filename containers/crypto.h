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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CRYPTO_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CRYPTO_H_

#include <memory>
#include <string>
#include <tuple>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/struct.pb.h"

namespace confidential_federated_compute {

// Class used to decrypt messages intended for this worker.
//
// All the KMS authorized decryption keys should be passed down to the
// Decryptor as part of its constructor.
//
// This class is threadsafe.
class Decryptor {
 public:
  // Constructs a new Decryptor.
  Decryptor(const std::vector<absl::string_view>& decryption_keys = {});

  // Decryptor is not copyable or moveable due to the use of
  // fcp::confidential_compute::MessageDecryptor.
  Decryptor(const Decryptor& other) = delete;
  Decryptor& operator=(const Decryptor& other) = delete;

  // Decrypts a record encrypted with the public key owned by this class.
  absl::StatusOr<std::string> DecryptBlob(
      const fcp::confidentialcompute::BlobMetadata& metadata,
      absl::string_view blob, absl::string_view key_id = "");

  // Unwraps a release token using the decryption keys owned by this class.
  absl::StatusOr<fcp::confidential_compute::UnwrappedReleaseToken>
  UnwrapReleaseToken(absl::string_view release_token) {
    return message_decryptor_.UnwrapReleaseToken(release_token);
  }

 private:
  fcp::confidential_compute::MessageDecryptor message_decryptor_;
};

// Wraps BoringSSL's HMAC-SHA256.
absl::StatusOr<std::string> KeyedHash(absl::string_view input,
                                      absl::string_view key);

// Returns a random 16 byte Blob ID
std::string RandomBlobId();

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CRYPTO_H_
