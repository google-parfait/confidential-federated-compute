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
#include "containers/crypto_test_utils.h"

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "openssl/aead.h"
#include "openssl/base.h"
#include "openssl/err.h"
#include "openssl/hpke.h"

namespace confidential_federated_compute::crypto_test_utils {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::crypto_internal::UnwrapSymmetricKey;
using ::fcp::confidential_compute::crypto_internal::WrapSymmetricKey;
using ::fcp::confidential_compute::crypto_internal::WrapSymmetricKeyResult;
using ::fcp::confidentialcompute::Record;

absl::StatusOr<Record> CreateRewrappedRecord(
    absl::string_view message, absl::string_view ciphertext_associated_data,
    absl::string_view recipient_public_key, absl::string_view nonce,
    absl::string_view reencryption_public_key) {
  MessageEncryptor encryptor;
  // Encrypt the symmetric key with the public key of an intermediary.
  bssl::ScopedEVP_HPKE_KEY intermediary_key;
  const EVP_HPKE_KEM* kem = EVP_hpke_x25519_hkdf_sha256();
  const EVP_HPKE_KDF* kdf = EVP_hpke_hkdf_sha256();
  const EVP_HPKE_AEAD* aead = EVP_hpke_aes_128_gcm();
  FCP_CHECK(EVP_HPKE_KEY_generate(intermediary_key.get(), kem) == 1);
  size_t public_key_len;
  std::string intermediary_public_key(EVP_HPKE_MAX_PUBLIC_KEY_LENGTH, '\0');
  FCP_CHECK(EVP_HPKE_KEY_public_key(
                intermediary_key.get(),
                reinterpret_cast<uint8_t*>(intermediary_public_key.data()),
                &public_key_len, intermediary_public_key.size()) == 1);
  intermediary_public_key.resize(public_key_len);
  FCP_ASSIGN_OR_RETURN(EncryptMessageResult encrypt_result,
                       encryptor.Encrypt(message, intermediary_public_key,
                                         ciphertext_associated_data));

  // Have the intermediary rewrap the symmetric key with the public key of the
  // final recipient.
  FCP_ASSIGN_OR_RETURN(
      std::string symmetric_key,
      UnwrapSymmetricKey(intermediary_key.get(), kdf, aead,
                         encrypt_result.encrypted_symmetric_key,
                         encrypt_result.encapped_key,
                         ciphertext_associated_data));

  FCP_ASSIGN_OR_RETURN(
      WrapSymmetricKeyResult rewrapped_symmetric_key_result,
      WrapSymmetricKey(kem, kdf, aead, symmetric_key, recipient_public_key,
                       absl::StrCat(reencryption_public_key, nonce)));

  Record record;
  record.mutable_hpke_plus_aead_data()->set_ciphertext(
      encrypt_result.ciphertext);
  record.mutable_hpke_plus_aead_data()->set_ciphertext_associated_data(
      std::string(ciphertext_associated_data));
  record.mutable_hpke_plus_aead_data()->set_encrypted_symmetric_key(
      rewrapped_symmetric_key_result.encrypted_symmetric_key);
  record.mutable_hpke_plus_aead_data()->set_encapsulated_public_key(
      rewrapped_symmetric_key_result.encapped_key);
  record.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_nonce(std::string(nonce));
  record.mutable_hpke_plus_aead_data()
      ->mutable_rewrapped_symmetric_key_associated_data()
      ->set_reencryption_public_key(std::string(reencryption_public_key));
  return record;
}

}  // namespace confidential_federated_compute::crypto_test_utils
