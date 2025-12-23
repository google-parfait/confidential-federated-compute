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
#include "containers/blob_metadata.h"

#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"

namespace confidential_federated_compute {

using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;

absl::StatusOr<std::string> GetKeyIdFromMetadata(
    const fcp::confidentialcompute::BlobMetadata& metadata) {
  // Returns an empty string for unencrypted payloads.
  if (metadata.has_unencrypted()) {
    return "";
  }

  // GetKeyId is only supported for KMS-enabled transforms.
  if (!metadata.hpke_plus_aead_data().has_kms_symmetric_key_associated_data()) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data is not present.");
  }

  fcp::confidentialcompute::BlobHeader blob_header;
  if (!blob_header.ParseFromString(metadata.hpke_plus_aead_data()
                                       .kms_symmetric_key_associated_data()
                                       .record_header())) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data.record_header() cannot be "
        "parsed to BlobHeader.");
  }

  if (blob_header.key_id().empty()) {
    return absl::InvalidArgumentError(
        "Parsed BlobHeader has an empty 'key_id'");
  }

  return blob_header.key_id();
}

}  // namespace confidential_federated_compute
