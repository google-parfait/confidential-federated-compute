// Copyright 2025 Google LLC.
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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MINIMUM_JAX_CONFIDENTIAL_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MINIMUM_JAX_CONFIDENTIAL_TRANSFORM_SERVER_H_

#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "cc/crypto/signing_key.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::minimum_jax {

class SimpleSession final : public confidential_federated_compute::Session {
 public:
  SimpleSession(std::vector<std::string> reencryption_keys,
                absl::string_view reencryption_policy_hash)
      : reencryption_keys_(std::move(reencryption_keys)),
        reencryption_policy_hash_(reencryption_policy_hash) {}

  absl::StatusOr<fcp::confidentialcompute::ConfigureResponse> Configure(
      fcp::confidentialcompute::ConfigureRequest request,
      Context& context) override {
    return fcp::confidentialcompute::ConfigureResponse();
  }
  // Adds a data blob from a given URI into the session and caches it.
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest request,
      std::string unencrypted_data, Context& context) override
      ABSL_LOCKS_EXCLUDED(mutex_);
  // No-op
  absl::StatusOr<fcp::confidentialcompute::CommitResponse> Commit(
      fcp::confidentialcompute::CommitRequest request,
      Context& context) override {
    return ToCommitResponse(absl::OkStatus());
  }
  // Perform computation on all cached data.
  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Finalize(
      fcp::confidentialcompute::FinalizeRequest request,
      fcp::confidentialcompute::BlobMetadata input_metadata,
      Context& context) override ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  absl::StatusOr<
      std::tuple<fcp::confidentialcompute::BlobMetadata, std::string>>
  EncryptSessionResult(absl::string_view plaintext);

  std::vector<std::string> data_ ABSL_GUARDED_BY(mutex_);
  absl::Mutex mutex_;
  std::vector<std::string> reencryption_keys_;
  std::string reencryption_policy_hash_;
};

class SimpleConfidentialTransform final
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  explicit SimpleConfidentialTransform(
      std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle,
      std::unique_ptr<oak::crypto::EncryptionKeyHandle> encryption_key_handle)
      : ConfidentialTransformBase(std::move(signing_key_handle),
                                  std::move(encryption_key_handle)) {}

 protected:
  // No-op
  absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override {
    return google::protobuf::Struct();
  }

  absl::Status StreamInitializeTransformWithKms(
      const google::protobuf::Any& configuration,
      const google::protobuf::Any& config_constraints,
      std::vector<std::string> reencryption_keys,
      absl::string_view reencryption_policy_hash) override {
    reencryption_keys_ = std::move(reencryption_keys);
    reencryption_policy_hash_ = reencryption_policy_hash;
    return absl::OkStatus();
  }

  // No-op
  absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override {
    return std::make_unique<SimpleSession>(reencryption_keys_,
                                           reencryption_policy_hash_);
  }

  absl::StatusOr<std::string> GetKeyId(
      const fcp::confidentialcompute::BlobMetadata& metadata) override;

 private:
  std::vector<std::string> reencryption_keys_;
  std::string reencryption_policy_hash_;
};
}  // namespace confidential_federated_compute::minimum_jax

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MINIMUM_JAX_CONFIDENTIAL_TRANSFORM_SERVER_H_