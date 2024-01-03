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
#include "containers/crypto.h"

#include <memory>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "openssl/rand.h"

namespace confidential_federated_compute {

constexpr size_t kNonceSize = 16;

using ::fcp::confidentialcompute::Record;

RecordDecryptor::RecordDecryptor(const google::protobuf::Any& configuration) {
  // TODO(nfallen): Generate a signature for the public key and configuration
  // once Oak libraries are available.
  absl::StatusOr<std::string> public_key = message_decryptor_.GetPublicKey();
  if (!public_key.ok()) {
    public_key_and_signature_ = public_key.status();
    return;
  }
  public_key_and_signature_ = PublicKeyAndSignature{
      .public_key = std::move(*public_key), .signature = ""};
}

absl::StatusOr<const PublicKeyAndSignature*>
RecordDecryptor::GetPublicKeyAndSignature() const {
  if (!public_key_and_signature_.ok()) {
    return public_key_and_signature_.status();
  }
  return &(public_key_and_signature_.value());
}

absl::StatusOr<std::string> RecordDecryptor::GenerateNonce() {
  bool inserted = false;
  std::string nonce(kNonceSize, '\0');
  while (!inserted) {
    // BoringSSL documentation says that it always returns 1 so we don't check
    // the return value.
    (void)RAND_bytes(reinterpret_cast<unsigned char*>(nonce.data()),
                     nonce.size());
    {
      absl::MutexLock l(&mutex_);
      auto pair = nonces_.insert(nonce);
      inserted = pair.second;
    }
  }
  return nonce;
}

absl::StatusOr<std::string> RecordDecryptor::DecryptRecord(
    const Record& record) {
  switch (record.kind_case()) {
    case Record::kUnencryptedData:
      return record.unencrypted_data();
    case Record::kHpkePlusAeadData:
      break;
    default:
      return absl::InvalidArgumentError(
          "Record to decrypt must contain unencrypted_data or "
          "rewrapped_symmetric_key_associated_data");
  }
  if (!record.hpke_plus_aead_data()
           .has_rewrapped_symmetric_key_associated_data()) {
    return absl::InvalidArgumentError(
        "Record to decrypt must contain "
        "rewrapped_symmetric_key_associated_data");
  }
  // Ensure the nonce is used exactly once.
  absl::string_view nonce = record.hpke_plus_aead_data()
                                .rewrapped_symmetric_key_associated_data()
                                .nonce();
  {
    absl::MutexLock l(&mutex_);
    if (!nonces_.erase(nonce)) {
      return absl::FailedPreconditionError(
          "Nonce not found on TEE. The same record may have already been "
          "decrypted by this TEE, or this TEE never generated a nonce for this "
          "record.");
    }
  }
  std::string associated_data =
      absl::StrCat(record.hpke_plus_aead_data()
                       .rewrapped_symmetric_key_associated_data()
                       .reencryption_public_key(),
                   nonce);
  return message_decryptor_.Decrypt(
      record.hpke_plus_aead_data().ciphertext(),
      record.hpke_plus_aead_data().ciphertext_associated_data(),
      record.hpke_plus_aead_data().encrypted_symmetric_key(), associated_data,
      record.hpke_plus_aead_data().encapsulated_public_key());
}

}  // namespace confidential_federated_compute
