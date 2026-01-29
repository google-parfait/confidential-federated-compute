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
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "openssl/err.h"
#include "openssl/evp.h"
#include "openssl/hmac.h"
#include "openssl/rand.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidential_compute::Ec2Cwt;
using ::fcp::confidential_compute::Ec2Key;
using ::fcp::confidential_compute::EcdsaP256R1SignatureVerifier;
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

// Creates a EcdsaP256R1SignatureVerifier from an EC2 key.
absl::StatusOr<EcdsaP256R1SignatureVerifier> CreateP256Verifier(
    const Ec2Key& key) {
  // Algorithm, curve, and key operation constants are defined in
  // https://www.iana.org/assignments/cose/cose.xhtml and RFC 9052 Table 5.
  if (key.algorithm != -7 /* ES256 */) {
    return absl::InvalidArgumentError("unsupported algorithm");
  } else if (key.curve != 1 /* P-256 */) {
    return absl::InvalidArgumentError("unsupported curve");
  } else if (!key.key_ops.empty() &&
             absl::c_find(key.key_ops, 2 /* verify */) == key.key_ops.end()) {
    return absl::InvalidArgumentError("key does not support verification");
  } else if (key.x.size() != 32) {
    return absl::InvalidArgumentError("unsupported key x coordinate");
  } else if (key.y.size() != 32) {
    return absl::InvalidArgumentError("unsupported key y coordinate");
  }
  // Uncompressed X9.62 keys have the form \x04 || x || y.
  return EcdsaP256R1SignatureVerifier::Create(
      absl::StrCat("\x04", key.x, key.y));
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

absl::StatusOr<BlobProvenance> VerifyBlobProvenance(
    absl::string_view data, absl::string_view signature,
    absl::string_view signing_key_endorsement, absl::string_view kms_public_key,
    absl::string_view expected_invocation_id) {
  // Create a signature verifier that uses the KMS public key.
  FCP_ASSIGN_OR_RETURN(Ec2Key kms_key, Ec2Key::Decode(kms_public_key));
  FCP_ASSIGN_OR_RETURN(EcdsaP256R1SignatureVerifier kms_verifier,
                       CreateP256Verifier(kms_key));

  // Parse and verify the signing key endorsement.
  FCP_ASSIGN_OR_RETURN(Ec2Cwt endorsement,
                       Ec2Cwt::Decode(signing_key_endorsement));
  if (endorsement.algorithm != kms_key.algorithm) {
    return absl::InvalidArgumentError(
        "signing key endorsement has wrong algorithm");
  }
  FCP_ASSIGN_OR_RETURN(
      std::string sig_structure,
      Ec2Cwt::GetSigStructureForVerifying(signing_key_endorsement, ""));
  FCP_RETURN_IF_ERROR(
      kms_verifier.Verify(sig_structure, endorsement.signature));
  if (!endorsement.public_key) {
    return absl::InvalidArgumentError(
        "signing key endorsement missing public key");
  } else if (!endorsement.transform_index) {
    return absl::InvalidArgumentError(
        "signing key endorsement missing transform index");
  } else if (endorsement.invocation_id != expected_invocation_id) {
    return absl::InvalidArgumentError(
        "signing key endorsement has wrong invocation ID");
  }
  FCP_ASSIGN_OR_RETURN(EcdsaP256R1SignatureVerifier endorsed_verifier,
                       CreateP256Verifier(*endorsement.public_key));

  // Verify the data's signature.
  FCP_RETURN_IF_ERROR(endorsed_verifier.Verify(data, signature));

  return BlobProvenance{
      .transform_index = *endorsement.transform_index,
      .dst_node_ids = std::move(endorsement.dst_node_ids),
  };
}

}  // namespace confidential_federated_compute
