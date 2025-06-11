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
#include "fcp/confidentialcompute/cose.h"

namespace confidential_federated_compute {

using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::BlobMetadata;

absl::StatusOr<BlobMetadata> EarliestExpirationTimeMetadata(
    const BlobMetadata& metadata, const BlobMetadata& other) {
  absl::Time metadata_time = absl::InfiniteFuture();
  absl::Time other_time = absl::InfiniteFuture();
  if (metadata.has_hpke_plus_aead_data()) {
    FCP_ASSIGN_OR_RETURN(
        OkpCwt metadata_cwt,
        OkpCwt::Decode(metadata.hpke_plus_aead_data()
                           .rewrapped_symmetric_key_associated_data()
                           .reencryption_public_key()));

    if (metadata_cwt.expiration_time.has_value()) {
      metadata_time = *metadata_cwt.expiration_time;
    }
  }
  if (other.has_hpke_plus_aead_data()) {
    FCP_ASSIGN_OR_RETURN(
        OkpCwt other_cwt,
        OkpCwt::Decode(other.hpke_plus_aead_data()
                           .rewrapped_symmetric_key_associated_data()
                           .reencryption_public_key()));
    if (other_cwt.expiration_time.has_value()) {
      other_time = *other_cwt.expiration_time;
    }
  }
  if (metadata_time < other_time) {
    return metadata;
  }
  return other;
}

}  // namespace confidential_federated_compute
