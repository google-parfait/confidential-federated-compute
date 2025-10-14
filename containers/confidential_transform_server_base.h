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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONFIDENTIAL_TRANSFORM_SERVER_BASE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONFIDENTIAL_TRANSFORM_SERVER_BASE_H_

#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "cc/crypto/encryption_key.h"
#include "cc/crypto/signing_key.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/confidentialcompute/nonce.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute {

using SessionStream =
    grpc::ServerReaderWriter<fcp::confidentialcompute::SessionResponse,
                             fcp::confidentialcompute::SessionRequest>;

// Base class that implements the ConfidentialTransform service protocol.
class ConfidentialTransformBase
    : public fcp::confidentialcompute::ConfidentialTransform::Service {
 public:
  grpc::Status StreamInitialize(
      grpc::ServerContext* context,
      grpc::ServerReader<fcp::confidentialcompute::StreamInitializeRequest>*
          reader,
      fcp::confidentialcompute::InitializeResponse* response) override;

  grpc::Status Session(grpc::ServerContext* context,
                       SessionStream* stream) override;

 protected:
  ConfidentialTransformBase(
      std::shared_ptr<oak::crypto::SigningKeyHandle> signing_key_handle,
      std::unique_ptr<oak::crypto::EncryptionKeyHandle> encryption_key_handle =
          nullptr)
      : oak_signing_key_handle_(std::move(signing_key_handle)),
        oak_encryption_key_handle_(std::move(encryption_key_handle)) {}

  // Initialize the transform with legacy ledger.
  // Either one of `StreamInitializeTransform` or
  // `StreamInitializeTransformWithKms` should be invoked but never both.
  virtual absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) = 0;
  // Initialize the transform when KMS is enabled for this worker.
  // Either one of `StreamInitializeTransform` or
  // `StreamInitializeTransformWithKms` should be invoked but never both.
  virtual absl::Status StreamInitializeTransformWithKms(
      const ::google::protobuf::Any& configuration,
      const ::google::protobuf::Any& config_constraints,
      std::vector<std::string> reencryption_keys,
      absl::string_view reencryption_policy_hash) {
    return absl::OkStatus();
  };
  // Handles a WriteConfigurationRequest that contains a blob or a chunk of a
  // blob used for container initialization. Must be implemented by a subclass.
  // The first WriteConfigurationRequest for each blob must contain the metadata
  // for the blob, while the last must have `commit` set to True.
  virtual absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) = 0;
  virtual absl::StatusOr<
      std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() = 0;

  // Retrieves the key_id from the BlobMetadata.
  virtual absl::StatusOr<std::string> GetKeyId(
      const fcp::confidentialcompute::BlobMetadata& metadata) = 0;

  // Transforms that have KMS enabled are meant to be used with the KMS. They
  // must
  // - Validate that they are configured with properties permitted by the KMS
  // - Track their own privacy budget.
  // - Release only encrypted results along with a release token.
  bool KmsEnabled() const { return kms_enabled_; }

  // Returns the authorized logical policy hashes for this container.
  absl::flat_hash_set<std::string>&
  GetAuthorizedLogicalPipelinePoliciesHashes() {
    return authorized_logical_pipeline_policies_hashes_;
  }

  // Returns the active key ids for this container.
  absl::flat_hash_set<std::string> GetActiveKeyIds() const {
    return active_key_ids_;
  }

  // Returns a pointer to the BlobDecryptor.
  absl::StatusOr<confidential_federated_compute::BlobDecryptor*>
  GetBlobDecryptor();

  // Returns the Oak SigningKeyHandle passed to the constructor.
  const std::shared_ptr<oak::crypto::SigningKeyHandle>& GetOakSigningKeyHandle()
      const {
    return oak_signing_key_handle_;
  }

 private:
  absl::Status StreamInitializeInternal(
      grpc::ServerReader<fcp::confidentialcompute::StreamInitializeRequest>*
          reader,
      fcp::confidentialcompute::InitializeResponse* response);

  absl::Status SessionImpl(SessionStream* stream);

  absl::Status HandleWrite(
      confidential_federated_compute::Session* session,
      fcp::confidentialcompute::WriteRequest request, absl::Cord blob_data,
      BlobDecryptor* blob_decryptor,
      std::optional<fcp::confidential_compute::NonceChecker>& nonce_checker,
      SessionStream* stream, Session::Context& context);

  absl::Status SetActiveKeyIds(
      const std::vector<absl::string_view>& decryption_keys,
      const std::vector<absl::string_view>& omitted_key_ids);

  absl::Mutex mutex_;
  // The mutex is used to protect the optional wrapping blob_decryptor_ and
  // session_tracker_ to ensure they are initialized, but the BlobDecryptor and
  // SessionTracker are themselves threadsafe.
  std::optional<confidential_federated_compute::BlobDecryptor> blob_decryptor_
      ABSL_GUARDED_BY(mutex_);
  std::optional<confidential_federated_compute::SessionTracker> session_tracker_
      ABSL_GUARDED_BY(mutex_);
  // TODO: Refactor ConfidentialTransformBase so it's not aware of either the
  // Ledger nor KMS. Future applications may not be based on either Ledger nor
  // KMS.
  bool kms_enabled_;
  std::shared_ptr<oak::crypto::SigningKeyHandle> oak_signing_key_handle_;
  std::unique_ptr<oak::crypto::EncryptionKeyHandle> oak_encryption_key_handle_;
  absl::flat_hash_set<std::string> authorized_logical_pipeline_policies_hashes_;
  // Tracks the keys ids that are still active i.e. not expired.
  absl::flat_hash_set<std::string> active_key_ids_;
};

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONFIDENTIAL_TRANSFORM_SERVER_BASE_H_
