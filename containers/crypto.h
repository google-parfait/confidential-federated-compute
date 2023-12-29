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

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"

namespace confidential_federated_compute {

struct PublicKeyAndSignature {
  // A public key that can be used to encrypt messages so they can be decrypted
  // by this container.
  std::string public_key;
  // A signature of the public key and the configuration of the container
  // application.
  std::string signature;
};

// Class used to decrypt Record protos that have been rewrapped for access by
// this container by the Ledger.
//
// This class is threadsafe.
class RecordDecryptor {
 public:
  explicit RecordDecryptor(const google::protobuf::Any& configuration);

  // RecordDecryptor is not copyable or moveable due to the use of
  // fcp::confidential_compute::MessageDecryptor.
  RecordDecryptor(const RecordDecryptor& other) = delete;
  RecordDecryptor& operator=(const RecordDecryptor& other) = delete;

  // Returns a pointer to a PublicKeyAndSignature which
  // demonstrates that this container received the configuration provided in the
  // constructor and generated the returned public key that can be used for
  // encrypting messages it can then decrypt.
  //
  // This class must outlive the pointer that is returned.
  //
  // If this method is called multiple times, the same public key and signature
  // will be returned. A caller wanting to update the configuration and generate
  // a new public key should create a new instance of this class.
  absl::StatusOr<const PublicKeyAndSignature*> GetPublicKeyAndSignature() const;

  // Generates and signs a nonce that can be used for identifying that a
  // certain record should be decrypted exactly once.
  absl::StatusOr<std::string> GenerateNonce();

  // Decrypts a record encrypted with the public key owned by this class.
  absl::StatusOr<std::string> DecryptRecord(
      const fcp::confidentialcompute::Record& record);

 private:
  absl::Mutex mutex_;
  absl::flat_hash_set<std::string> nonces_ ABSL_GUARDED_BY(mutex_);
  fcp::confidential_compute::MessageDecryptor message_decryptor_;
  absl::StatusOr<PublicKeyAndSignature> public_key_and_signature_;
};

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CRYPTO_H_
