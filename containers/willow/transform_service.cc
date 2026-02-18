// Copyright 2026 Google LLC.
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

#include "transform_service.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cc/crypto/encryption_key.h"
#include "cc/crypto/signing_key.h"
#include "containers/blob_metadata.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "willow_session.h"

namespace confidential_federated_compute::willow {

TransformService::TransformService(
    std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle,
    std::unique_ptr<oak::crypto::EncryptionKeyHandle> encryption_key_handle)
    : ConfidentialTransformBase(std::move(signing_key_handle),
                                std::move(encryption_key_handle)) {};

absl::Status TransformService::StreamInitializeTransform(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints) {
  return absl::OkStatus();
}

absl::Status TransformService::ReadWriteConfigurationRequest(
    const fcp::confidentialcompute::WriteConfigurationRequest&
        write_configuration) {
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Session>> TransformService::CreateSession() {
  return std::make_unique<WillowSession>();
};

absl::StatusOr<std::string> TransformService::GetKeyId(
    const fcp::confidentialcompute::BlobMetadata& metadata) {
  return GetKeyIdFromMetadata(metadata);
}

}  // namespace confidential_federated_compute::willow
