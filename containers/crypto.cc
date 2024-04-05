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
#include "absl/strings/string_view.h"
#include "fcp/base/compression.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "google/protobuf/struct.pb.h"
#include "grpcpp/client_context.h"
#include "openssl/rand.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"
#include "proto/containers/orchestrator_crypto.pb.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidentialcompute::Record;
using ::oak::containers::v1::OrchestratorCrypto;

constexpr size_t kNonceSize = 16;

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
    absl::string_view input, Record::CompressionType compression_type) {
  switch (compression_type) {
    case Record::COMPRESSION_TYPE_NONE:
      return std::string(input);

    case Record::COMPRESSION_TYPE_GZIP: {
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

RecordDecryptor::RecordDecryptor(const google::protobuf::Any& configuration,
                                 OrchestratorCrypto::StubInterface& stub,
                                 google::protobuf::Struct config_properties)
    : message_decryptor_(std::move(config_properties)),
      signed_public_key_(message_decryptor_.GetPublicKey(
          [&stub](absl::string_view message) {
            return SignWithOrchestrator(stub, message);
          },
          kAlgorithmES256)) {}

absl::StatusOr<absl::string_view> RecordDecryptor::GetPublicKey() const {
  if (!signed_public_key_.ok()) {
    return signed_public_key_.status();
  }
  return *signed_public_key_;
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
      return Decompress(record.unencrypted_data(), record.compression_type());
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
  FCP_ASSIGN_OR_RETURN(
      std::string decrypted,
      message_decryptor_.Decrypt(
          record.hpke_plus_aead_data().ciphertext(),
          record.hpke_plus_aead_data().ciphertext_associated_data(),
          record.hpke_plus_aead_data().encrypted_symmetric_key(),
          associated_data,
          record.hpke_plus_aead_data().encapsulated_public_key()));
  return Decompress(decrypted, record.compression_type());
}

}  // namespace confidential_federated_compute
