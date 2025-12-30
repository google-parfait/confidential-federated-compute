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
#include <tuple>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "openssl/aead.h"
#include "openssl/base.h"
#include "openssl/err.h"
#include "openssl/hpke.h"
#include "proto/crypto/crypto.pb.h"

namespace confidential_federated_compute::crypto_test_utils {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidential_compute::crypto_internal::CoseAlgorithm;
using ::fcp::confidential_compute::crypto_internal::CoseEllipticCurve;
using ::fcp::confidential_compute::crypto_internal::UnwrapSymmetricKey;
using ::fcp::confidential_compute::crypto_internal::WrapSymmetricKey;
using ::fcp::confidential_compute::crypto_internal::WrapSymmetricKeyResult;
using ::fcp::confidentialcompute::BlobMetadata;
using ::testing::Return;

const int64_t kHpkeBaseX25519Sha256Aes128Gcm = -65537;
const int64_t kX25519 = 4;

std::pair<std::string, std::string> GenerateKeyPair(std::string key_id) {
  bssl::ScopedEVP_HPKE_KEY key;
  EVP_HPKE_KEY_generate(key.get(), EVP_hpke_x25519_hkdf_sha256());
  size_t key_len;
  std::string raw_public_key(EVP_HPKE_MAX_PUBLIC_KEY_LENGTH, '\0');
  EVP_HPKE_KEY_public_key(key.get(),
                          reinterpret_cast<uint8_t*>(raw_public_key.data()),
                          &key_len, raw_public_key.size());
  raw_public_key.resize(key_len);
  std::string raw_private_key(EVP_HPKE_MAX_PRIVATE_KEY_LENGTH, '\0');
  EVP_HPKE_KEY_private_key(key.get(),
                           reinterpret_cast<uint8_t*>(raw_private_key.data()),
                           &key_len, raw_private_key.size());
  raw_private_key.resize(key_len);
  absl::StatusOr<std::string> public_key =
      OkpKey{
          .key_id = key_id,
          .algorithm = kHpkeBaseX25519Sha256Aes128Gcm,
          .curve = kX25519,
          .x = raw_public_key,
      }
          .Encode();
  CHECK_OK(public_key);
  absl::StatusOr<std::string> private_key =
      OkpKey{
          .key_id = key_id,
          .algorithm = kHpkeBaseX25519Sha256Aes128Gcm,
          .curve = kX25519,
          .d = raw_private_key,
      }
          .Encode();
  CHECK_OK(private_key);
  return {public_key.value(), private_key.value()};
}

MockSigningKeyHandle::MockSigningKeyHandle() {
  ON_CALL(*this, Sign)
      .WillByDefault(Return(oak::crypto::v1::Signature::default_instance()));
}

}  // namespace confidential_federated_compute::crypto_test_utils
