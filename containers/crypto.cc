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
#include "google/protobuf/struct.pb.h"
#include "grpcpp/client_context.h"
#include "openssl/err.h"
#include "openssl/evp.h"
#include "openssl/hmac.h"
#include "openssl/rand.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"
#include "proto/containers/orchestrator_crypto.pb.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::oak::containers::v1::OrchestratorCrypto;

constexpr size_t kNonceSize = 16;
constexpr size_t kBlobIdSize = 16;

// See https://www.iana.org/assignments/cose/cose.xhtml.
constexpr int64_t kAlgorithmES256 = -7;

absl::StatusOr<std::string> SignWithOrchestrator(
    OrchestratorCrypto::StubInterface& stub, absl::string_view message) {
  grpc::ClientContext context;
  oak::containers::v1::SignRequest request;
  request.set_key_origin(oak::containers::v1::KeyOrigin::INSTANCE);
  request.set_message(std::string(message));
  oak::containers::v1::SignResponse response;
  if (auto status = stub.Sign(&context, request, &response); !status.ok()) {
    return fcp::base::FromGrpcStatus(std::move(status));
  }
  return std::move(*response.mutable_signature()->mutable_signature());
}

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

BlobDecryptor::BlobDecryptor(
    OrchestratorCrypto::StubInterface& stub,
    google::protobuf::Struct config_properties,
    const std::vector<absl::string_view>& decryption_keys,
    std::optional<absl::flat_hash_set<std::string>>
        authorized_logical_pipeline_policies_hashes)
    : message_decryptor_(std::move(config_properties), decryption_keys),
      signed_public_key_(message_decryptor_.GetPublicKey(
          [&stub](absl::string_view message) {
            return SignWithOrchestrator(stub, message);
          },
          kAlgorithmES256)),
      authorized_logical_pipeline_policies_hashes_(
          authorized_logical_pipeline_policies_hashes) {}

absl::StatusOr<absl::string_view> BlobDecryptor::GetPublicKey() const {
  if (!signed_public_key_.ok()) {
    return signed_public_key_.status();
  }
  return *signed_public_key_;
}

absl::StatusOr<std::string> BlobDecryptor::DecryptBlob(
    const BlobMetadata& metadata, absl::string_view blob) {
  std::string decrypted;
  switch (metadata.encryption_metadata_case()) {
    case BlobMetadata::kUnencrypted:
      return Decompress(blob, metadata.compression_type());
    case BlobMetadata::kHpkePlusAeadData:
      if (metadata.hpke_plus_aead_data()
              .has_rewrapped_symmetric_key_associated_data()) {
        std::string associated_data =
            absl::StrCat(metadata.hpke_plus_aead_data()
                             .rewrapped_symmetric_key_associated_data()
                             .reencryption_public_key(),
                         metadata.hpke_plus_aead_data()
                             .rewrapped_symmetric_key_associated_data()
                             .nonce());
        FCP_ASSIGN_OR_RETURN(
            decrypted,
            message_decryptor_.Decrypt(
                blob,
                metadata.hpke_plus_aead_data().ciphertext_associated_data(),
                metadata.hpke_plus_aead_data().encrypted_symmetric_key(),
                associated_data,
                metadata.hpke_plus_aead_data().encapsulated_public_key()));
      } else if (metadata.hpke_plus_aead_data()
                     .has_kms_symmetric_key_associated_data()) {
        BlobHeader blob_header;
        if (!blob_header.ParseFromString(
                metadata.hpke_plus_aead_data()
                    .kms_symmetric_key_associated_data()
                    .record_header())) {
          return absl::InvalidArgumentError(
              "kms_symmetric_key_associated_data.record_header() cannot be "
              "parsed to BlobHeader.");
        }
        if (authorized_logical_pipeline_policies_hashes_.has_value() &&
            !authorized_logical_pipeline_policies_hashes_->empty() &&
            !authorized_logical_pipeline_policies_hashes_->contains(
                blob_header.access_policy_sha256())) {
          return absl::InvalidArgumentError(
              "BlobHeader.access_policy_sha256 does not match any "
              "authorized_logical_pipeline_policies_hashes returned by KMS.");
        }
        FCP_ASSIGN_OR_RETURN(
            decrypted,
            message_decryptor_.Decrypt(
                blob,
                metadata.hpke_plus_aead_data().ciphertext_associated_data(),
                metadata.hpke_plus_aead_data().encrypted_symmetric_key(),
                metadata.hpke_plus_aead_data()
                    .kms_symmetric_key_associated_data()
                    .record_header(),
                metadata.hpke_plus_aead_data().encapsulated_public_key(),
                blob_header.key_id()));
      } else {
        return absl::InvalidArgumentError(
            "Blob to decrypt must contain either "
            "rewrapped_symmetric_key_associated_data or "
            "kms_symmetric_key_associated_data");
      }
      break;
    default:
      return absl::InvalidArgumentError(
          "Blob to decrypt must contain unencrypted_data,  "
          "rewrapped_symmetric_key_associated_data or "
          "kms_symmetric_key_associated_data");
  }

  return Decompress(decrypted, metadata.compression_type());
}

absl::StatusOr<std::tuple<BlobMetadata, std::string>>
BlobEncryptor::EncryptBlob(absl::string_view plaintext,
                           absl::string_view public_key,
                           absl::string_view access_policy_sha256,
                           uint32_t access_policy_node_id) {
  FCP_ASSIGN_OR_RETURN(OkpCwt cwt, OkpCwt::Decode(public_key));
  if (!cwt.public_key.has_value()) {
    return absl::InvalidArgumentError("public key is invalid");
  }
  BlobHeader header;

  std::string blob_id(kBlobIdSize, '\0');
  // BoringSSL documentation says that it always returns 1 so we don't check
  // the return value.
  (void)RAND_bytes(reinterpret_cast<unsigned char*>(blob_id.data()),
                   blob_id.size());
  header.set_blob_id(blob_id);
  header.set_key_id(cwt.public_key->key_id);
  header.set_access_policy_node_id(access_policy_node_id);
  header.set_access_policy_sha256(std::string(access_policy_sha256));
  std::string serialized_header = header.SerializeAsString();

  FCP_ASSIGN_OR_RETURN(
      EncryptMessageResult encrypted_message,
      message_encryptor_.Encrypt(plaintext, public_key, serialized_header));

  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(encrypted_message.ciphertext.size());
  BlobMetadata::HpkePlusAeadMetadata* hpke_plus_aead_metadata =
      metadata.mutable_hpke_plus_aead_data();
  hpke_plus_aead_metadata->set_ciphertext_associated_data(serialized_header);
  hpke_plus_aead_metadata->set_encrypted_symmetric_key(
      std::move(encrypted_message.encrypted_symmetric_key));
  hpke_plus_aead_metadata->set_encapsulated_public_key(
      std::move(encrypted_message.encapped_key));
  hpke_plus_aead_metadata->mutable_ledger_symmetric_key_associated_data()
      ->set_record_header(std::move(serialized_header));
  return std::make_tuple(std::move(metadata),
                         std::move(encrypted_message.ciphertext));
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
