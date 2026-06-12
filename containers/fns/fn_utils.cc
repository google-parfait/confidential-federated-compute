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
#include "containers/fns/fn_utils.h"

#include <string>

#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::fns {

std::string GetBlobId(const fcp::confidentialcompute::BlobMetadata& metadata) {
  switch (metadata.encryption_metadata_case()) {
    case fcp::confidentialcompute::BlobMetadata::kUnencrypted:
      return metadata.unencrypted().blob_id();
    case fcp::confidentialcompute::BlobMetadata::kHpkePlusAeadData:
      return metadata.hpke_plus_aead_data().blob_id();
    default:
      return "";
  }
}

}  // namespace confidential_federated_compute::fns
