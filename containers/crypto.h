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
#include "cc/crypto/signing_key.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/struct.pb.h"

namespace confidential_federated_compute {

// Class used to decrypt blobs.
//
// If used with the legacy ledger, blobs should have been rewrapped for access
// by this container by the Ledger. If used with KMS, all the KMS authorized
// decryption keys should be passed down to the BlobDecryptor as part of its
// constructor.
//
// This class is threadsafe.
class BlobDecryptor {
 public:
  // Constructs a new BlobDecryptor.
  //
  // Optional configuration properties may be supplied in a Struct to document
  // any important configuration that should be verifiable during attestation by
  // the ledger; see ApplicationMatcher.config_properties in
  // https://github.com/google/federated-compute/blob/main/fcp/protos/confidentialcompute/access_policy.proto.
  // Configuration properties are not required to be passed when using KMS since
  // validation of the config constraints is performed by the worker itself.
  BlobDecryptor(oak::crypto::SigningKeyHandle& signing_key,
                const google::protobuf::Struct& config_properties = {},
                const std::vector<absl::string_view>& decryption_keys = {});

  // BlobDecryptor is not copyable or moveable due to the use of
  // fcp::confidential_compute::MessageDecryptor.
  BlobDecryptor(const BlobDecryptor& other) = delete;
  BlobDecryptor& operator=(const BlobDecryptor& other) = delete;

  // Returns a string_view encoding a public key and signature (represented as a
  // signed CWT). This key can be used for encrypting messages that can be
  // decrypted by this object. The CWT also contains claims for any
  // configuration properties supplied when the BlobDecryptor was constructed.
  //
  // This class must outlive the string_view that is returned.
  //
  // If this method is called multiple times, the same public key and signature
  // will be returned. A caller wanting to update the configuration and generate
  // a new public key should create a new instance of this class.
  absl::StatusOr<absl::string_view> GetPublicKey() const;

  // Decrypts a record encrypted with the public key owned by this class.
  absl::StatusOr<std::string> DecryptBlob(
      const fcp::confidentialcompute::BlobMetadata& metadata,
      absl::string_view blob, absl::string_view key_id = "");

 private:
  fcp::confidential_compute::MessageDecryptor message_decryptor_;
  absl::StatusOr<std::string> signed_public_key_;
};

// Class used to create an encrypted blob with a symmetric key that can be
// decrypted by the Ledger.
//
// This class is threadsafe.
class BlobEncryptor {
 public:
  absl::StatusOr<
      std::tuple<fcp::confidentialcompute::BlobMetadata, std::string>>
  EncryptBlob(absl::string_view plaintext, absl::string_view public_key,
              absl::string_view access_policy_sha256,
              uint32_t access_policy_node_id);

 private:
  const fcp::confidential_compute::MessageEncryptor message_encryptor_;
};

// Wraps BoringSSL's HMAC-SHA256.
absl::StatusOr<std::string> KeyedHash(absl::string_view input,
                                      absl::string_view key);

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CRYPTO_H_
