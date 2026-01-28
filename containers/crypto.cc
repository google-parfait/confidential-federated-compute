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

#include <cstdint>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "fcp/base/compression.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "openssl/err.h"
#include "openssl/evp.h"
#include "openssl/hmac.h"
#include "openssl/rand.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;

constexpr size_t kBlobIdSize = 16;

absl::StatusOr<std::string> Decompress(
    absl::string_view input, BlobMetadata::CompressionType compression_type) {
  switch (compression_type) {
    case BlobMetadata::COMPRESSION_TYPE_NONE:
      return std::string(input);

    case BlobMetadata::COMPRESSION_TYPE_GZIP: {
      FCP_ASSIGN_OR_RETURN(absl::Cord decompressed,
                           fcp::UncompressWithGzip(input));
      return std::string(decompressed);
    }

    default:
      return absl::InvalidArgumentError(
          absl::StrCat("unsupported compression type ", compression_type));
  }
}

}  // namespace

std::string RandomBlobId() {
  std::string blob_id(kBlobIdSize, '\0');
  // BoringSSL documentation says that it always returns 1 so we don't check
  // the return value.
  (void)RAND_bytes(reinterpret_cast<unsigned char*>(blob_id.data()),
                   blob_id.size());
  return blob_id;
}

Decryptor::Decryptor(const std::vector<absl::string_view>& decryption_keys)
    : message_decryptor_(decryption_keys) {}

absl::StatusOr<std::string> Decryptor::DecryptBlob(const BlobMetadata& metadata,
                                                   absl::string_view blob,
                                                   absl::string_view key_id) {
  std::string decrypted;
  switch (metadata.encryption_metadata_case()) {
    case BlobMetadata::kUnencrypted:
      return Decompress(blob, metadata.compression_type());
    case BlobMetadata::kHpkePlusAeadData: {
      if (!metadata.hpke_plus_aead_data()
               .has_kms_symmetric_key_associated_data()) {
        return absl::InvalidArgumentError(
            "Blob to decrypt must contain kms_symmetric_key_associated_data");
      }
      FCP_ASSIGN_OR_RETURN(
          decrypted,
          message_decryptor_.Decrypt(
              blob, metadata.hpke_plus_aead_data().ciphertext_associated_data(),
              metadata.hpke_plus_aead_data().encrypted_symmetric_key(),
              metadata.hpke_plus_aead_data()
                  .kms_symmetric_key_associated_data()
                  .record_header(),
              metadata.hpke_plus_aead_data().encapsulated_public_key(),
              key_id));
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Blob to decrypt must contain unencrypted_data or "
          "kms_symmetric_key_associated_data");
  }

  return Decompress(decrypted, metadata.compression_type());
}

absl::StatusOr<std::string> KeyedHash(absl::string_view input,
                                      absl::string_view key) {
  // Calculate the HMAC-SHA256 hash using BoringSSL.
  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int hash_len;
  if (HMAC(EVP_sha256(), reinterpret_cast<const unsigned char*>(key.data()),
           key.size(), reinterpret_cast<const unsigned char*>(input.data()),
           input.size(), hash, &hash_len) == nullptr) {
    unsigned long err = ERR_get_error();
    return absl::InternalError(
        absl::StrFormat("HMAC failed: %s", ERR_error_string(err, nullptr)));
  }

  // Return the hash as a std::string.
  return std::string(reinterpret_cast<char*>(hash), hash_len);
}

}  // namespace confidential_federated_compute
