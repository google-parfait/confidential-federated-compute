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

// This file contains utilities related to BlobMetadata.
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_BLOB_METADATA_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_BLOB_METADATA_H_

#include "absl/status/statusor.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"

namespace confidential_federated_compute {

// Returns the BlobMetadata with the earliest expiration time encoded in its
// reencryption public key.
//
// Metadata for unencrypted data and an OkpCwt without an expiration time is
// considered to have an expiration time of infinity. Returns an error if either
// of the reencryption public keys cannot be decoded.
absl::StatusOr<fcp::confidentialcompute::BlobMetadata>
EarliestExpirationTimeMetadata(
    const fcp::confidentialcompute::BlobMetadata& metadata,
    const fcp::confidentialcompute::BlobMetadata& other);

// Translate the metadata in a given Record into a BlobMetadata.
fcp::confidentialcompute::BlobMetadata GetBlobMetadataFromRecord(
    const fcp::confidentialcompute::Record& record);

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_BLOB_METADATA_H_
