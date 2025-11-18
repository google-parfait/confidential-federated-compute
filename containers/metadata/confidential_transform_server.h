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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_METADATA_CONFIDENTIAL_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_METADATA_CONFIDENTIAL_TRANSFORM_SERVER_H_

#include <string>
#include <vector>

#include "containers/confidential_transform_server_base.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/tee_payload_metadata.pb.h"

namespace confidential_federated_compute::metadata {

// ConfidentialTransform service for metadata labeling.
class MetadataConfidentialTransform final
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  MetadataConfidentialTransform(
      std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle,
      std::unique_ptr<oak::crypto::EncryptionKeyHandle> encryption_key_handle =
          nullptr)
      : confidential_federated_compute::ConfidentialTransformBase(
            std::move(signing_key_handle), std::move(encryption_key_handle)) {};

 private:
  absl::Status StreamInitializeTransformWithKms(
      const google::protobuf::Any& configuration,
      const google::protobuf::Any& config_constraints,
      std::vector<std::string> reencryption_keys,
      absl::string_view reencryption_policy_hash) override;
  absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override;
  absl::StatusOr<std::string> GetKeyId(
      const fcp::confidentialcompute::BlobMetadata& metadata) override;

  absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override {
    return absl::InternalError(
        "Metadata container must be initialized with KMS.");
  }
  absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) override {
    return absl::InternalError(
        "Metadata container does not support WriteConfigurationRequests.");
  }

  std::optional<const fcp::confidentialcompute::MetadataContainerConfig>
      config_;
};

}  // namespace confidential_federated_compute::metadata

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_METADATA_CONFIDENTIAL_TRANSFORM_SERVER_H_
