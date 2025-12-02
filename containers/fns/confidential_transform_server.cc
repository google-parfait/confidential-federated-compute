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
#include "containers/fns/confidential_transform_server.h"

#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/fns/fn_factory.h"
#include "containers/session.h"
#include "fcp/base/monitoring.h"

namespace confidential_federated_compute::fns {

absl::Status FnConfidentialTransform::StreamInitializeTransformWithKms(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    std::vector<std::string> reencryption_keys,
    absl::string_view reencryption_policy_hash) {
  absl::WriterMutexLock l(&fn_factory_mutex_);
  if (fn_factory_.has_value()) {
    return absl::FailedPreconditionError("Fn container already initialized.");
  }
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<FnFactory> fn_factory,
                       fn_factory_provider_(configuration, config_constraints));
  fn_factory_.emplace(std::move(fn_factory));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
FnConfidentialTransform::CreateSession() {
  absl::ReaderMutexLock l(&fn_factory_mutex_);
  if (!fn_factory_.has_value()) {
    return absl::FailedPreconditionError(
        "Fn container must be initialized before creating Fns.");
  }
  return (*fn_factory_)->CreateFn();
}

absl::StatusOr<std::string> FnConfidentialTransform::GetKeyId(
    const fcp::confidentialcompute::BlobMetadata& metadata) {
  if (!metadata.hpke_plus_aead_data().has_kms_symmetric_key_associated_data()) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data is not present.");
  }

  // Parse the BlobHeader to get the access policy hash and key ID.
  fcp::confidentialcompute::BlobHeader blob_header;
  if (!blob_header.ParseFromString(metadata.hpke_plus_aead_data()
                                       .kms_symmetric_key_associated_data()
                                       .record_header())) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data.record_header() cannot be "
        "parsed to BlobHeader.");
  }

  // Verify that the access policy hash matches one of the authorized
  // logical pipeline policy hashes returned by KMS before returning the key
  // ID.
  if (!GetAuthorizedLogicalPipelinePoliciesHashes().contains(
          blob_header.access_policy_sha256())) {
    return absl::InvalidArgumentError(
        "BlobHeader.access_policy_sha256 does not match any "
        "authorized_logical_pipeline_policies_hashes returned by KMS.");
  }
  return blob_header.key_id();
}

}  // namespace confidential_federated_compute::fns
