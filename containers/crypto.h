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
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "google/protobuf/struct.pb.h"
#include "proto/containers/orchestrator_crypto.grpc.pb.h"

namespace confidential_federated_compute {

// Class used to track a session-level nonce and blob counter.
//
// This class is not threadsafe.
class SessionNonceTracker {
 public:
  SessionNonceTracker();
  // Checks that the blob-level nonce matches the session-level nonce and blob
  // counter. If so, increments the counter. If the blob is unencrypted, always
  // returns OK and doesn't increment the blob counter.
  absl::Status CheckBlobNonce(
      const fcp::confidentialcompute::BlobMetadata& blob_metadata);

  std::string GetSessionNonce() { return session_nonce_; }

 private:
  std::string session_nonce_;
  uint32_t counter_ = 0;
};

// Class used to decrypt blobs that have been rewrapped for access by
// this container by the Ledger.
//
// Unlike RecordDecryptor, this class does not track nonces to ensure that each
// blob can be decrypted once. This class is threadsafe.
class BlobDecryptor {
 public:
  // Constructs a new BlobDecryptor.
  //
  // Optional configuration properties may be supplied in a Struct to document
  // any important configuration that should be verifiable during attestation;
  // see ApplicationMatcher.config_properties in
  // https://github.com/google/federated-compute/blob/main/fcp/protos/confidentialcompute/access_policy.proto.
  BlobDecryptor(oak::containers::v1::OrchestratorCrypto::StubInterface& stub,
                google::protobuf::Struct config_properties = {});

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
      absl::string_view blob);

 private:
  fcp::confidential_compute::MessageDecryptor message_decryptor_;
  absl::StatusOr<std::string> signed_public_key_;
};

// Class used to create an encrypted Record with a symmetric key that can be
// decrypted by the Ledger.
//
// This class is threadsafe.
class RecordEncryptor {
 public:
  absl::StatusOr<fcp::confidentialcompute::Record> EncryptRecord(
      absl::string_view plaintext, absl::string_view public_key,
      absl::string_view access_policy_sha256, uint32_t access_policy_node_id);

 private:
  const fcp::confidential_compute::MessageEncryptor message_encryptor_;
};

// Class used to decrypt Record protos that have been rewrapped for access by
// this container by the Ledger.
//
// This class is threadsafe.
// TODO: Remove this class once we transition the existing containers to the new
// untrusted to TEE API.
class RecordDecryptor {
 public:
  // Constructs a new RecordDecryptor.
  //
  // Optional configuration properties may be supplied in a Struct to document
  // any important configuration that should be verifiable during attestation;
  // see ApplicationMatcher.config_properties in
  // https://github.com/google/federated-compute/blob/main/fcp/protos/confidentialcompute/access_policy.proto.
  RecordDecryptor(oak::containers::v1::OrchestratorCrypto::StubInterface& stub,
                  google::protobuf::Struct config_properties = {});

  // RecordDecryptor is not copyable or moveable due to the use of
  // fcp::confidential_compute::MessageDecryptor.
  RecordDecryptor(const RecordDecryptor& other) = delete;
  RecordDecryptor& operator=(const RecordDecryptor& other) = delete;

  // Returns a string_view encoding a public key and signature (represented as a
  // signed CWT). This key can be used for encrypting messages that can be
  // decrypted by this object. The CWT also contains claims for any
  // configuration properties supplied when the RecordDecryptor was constructed.
  //
  // This class must outlive the string_view that is returned.
  //
  // If this method is called multiple times, the same public key and signature
  // will be returned. A caller wanting to update the configuration and generate
  // a new public key should create a new instance of this class.
  absl::StatusOr<absl::string_view> GetPublicKey() const;

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
  absl::StatusOr<std::string> signed_public_key_;
};

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CRYPTO_H_
