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
#include "containers/metadata/confidential_transform_server.h"

namespace confidential_federated_compute::metadata {

namespace {
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::MetadataContainerConfig;

}  // namespace

absl::Status MetadataConfidentialTransform::StreamInitializeTransformWithKms(
    const google::protobuf::Any& /*configuration*/,
    const google::protobuf::Any& config_constraints,
    std::vector<std::string> /*reencryption_keys*/,
    absl::string_view /*reencryption_policy_hash*/) {
  // We expect to find the configuration that describes what metadata to
  // generate in the config_constraints. We don't use the untrusted
  // `configuration` parameter.
  MetadataContainerConfig config;
  if (!config_constraints.UnpackTo(&config)) {
    return absl::InvalidArgumentError(
        "Config constraints cannot be unpacked to MetadataContainerConfig.");
  }
  config_.emplace(std::move(config));
  // We don't use the reencryption parameters since the metadata container only
  // generates unencrypted outputs.
  return absl::OkStatus();
}

absl::StatusOr<std::string> MetadataConfidentialTransform::GetKeyId(
    const fcp::confidentialcompute::BlobMetadata& metadata) {
  if (!metadata.hpke_plus_aead_data().has_kms_symmetric_key_associated_data()) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data is not present.");
  }

  // Parse the BlobHeader to get the access policy hash and key ID.
  BlobHeader blob_header;
  if (!blob_header.ParseFromString(metadata.hpke_plus_aead_data()
                                       .kms_symmetric_key_associated_data()
                                       .record_header())) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data.record_header() cannot be "
        "parsed to BlobHeader.");
  }

  // Verify that the access policy hash matches one of the authorized
  // logical pipeline policy hashes returned by KMS before returning the key ID.
  if (!GetAuthorizedLogicalPipelinePoliciesHashes().contains(
          blob_header.access_policy_sha256())) {
    return absl::InvalidArgumentError(
        "BlobHeader.access_policy_sha256 does not match any "
        "authorized_logical_pipeline_policies_hashes returned by KMS.");
  }
  return blob_header.key_id();
}

absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
MetadataConfidentialTransform::CreateSession() {
  return absl::UnimplementedError("Not implemented yet.");
}

}  // namespace confidential_federated_compute::metadata
