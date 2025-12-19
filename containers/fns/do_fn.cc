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
#include "containers/fns/do_fn.h"

#include <string>

#include "absl/status/statusor.h"
#include "containers/fns/fn.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::fns {

namespace {
// Returns the blob ID from the BlobMetadata if it exists, otherwise returns an
// empty string.
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
}  // namespace

absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> DoFn::Write(
    fcp::confidentialcompute::WriteRequest write_request,
    std::string unencrypted_data, Context& context) {
  size_t unencrypted_data_size = unencrypted_data.size();
  KV input;
  input.data = std::move(unencrypted_data);
  input.blob_id = GetBlobId(write_request.first_request_metadata());
  input.key = std::move(write_request.first_request_configuration());
  FCP_RETURN_IF_ERROR(Do(std::move(input), context));

  fcp::confidentialcompute::WriteFinishedResponse response;
  response.set_committed_size_bytes(unencrypted_data_size);
  return response;
}
}  // namespace confidential_federated_compute::fns