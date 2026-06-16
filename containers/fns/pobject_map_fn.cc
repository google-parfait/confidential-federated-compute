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
#include "containers/fns/pobject_map_fn.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "containers/fns/fn_utils.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {

absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse>
PObjectMapFn::Write(fcp::confidentialcompute::WriteRequest write_request,
                    std::string unencrypted_data, Context& context) {
  size_t data_size = unencrypted_data.size();
  Session::KV input;
  input.data = std::move(unencrypted_data);
  input.blob_id = GetBlobId(write_request.first_request_metadata());
  input.key = std::move(write_request.first_request_configuration());
  accumulated_inputs_.push_back(std::move(input));

  fcp::confidentialcompute::WriteFinishedResponse response;
  response.set_committed_size_bytes(data_size);
  return response;
}

absl::StatusOr<fcp::confidentialcompute::CommitResponse> PObjectMapFn::Commit(
    fcp::confidentialcompute::CommitRequest commit_request, Context& context) {
  // Move accumulated data out and clear defensively. This ensures clean state
  // even if Map() returns an error and the Fn is somehow reused.
  auto inputs = std::move(accumulated_inputs_);
  accumulated_inputs_.clear();
  int32_t num_inputs = inputs.size();
  ABSL_RETURN_IF_ERROR(Map(std::move(*commit_request.mutable_configuration()),
                           std::move(inputs), context));
  fcp::confidentialcompute::CommitResponse response;
  response.mutable_stats()->set_num_inputs_committed(num_inputs);
  return response;
}

}  // namespace confidential_federated_compute::fns
