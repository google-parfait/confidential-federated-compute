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
#include "containers/fns/global_reduce_fn.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "containers/common/privacy_id_utils.h"
#include "containers/fns/fn.h"
#include "containers/fns/fn_utils.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {

namespace {

using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::ProtectedMetadata;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;

}  // namespace

absl::StatusOr<WriteFinishedResponse> GlobalReduceFn::Write(
    WriteRequest write_request, std::string unencrypted_data,
    ProtectedMetadata protected_metadata, Context& context) {
  if (committed_) {
    return absl::FailedPreconditionError(
        "GlobalReduceFn doesn't accept Write() after Commit().");
  }

  size_t data_size = unencrypted_data.size();
  Element input;
  input.kv.data = std::move(unencrypted_data);
  input.kv.blob_id =
      GetAuthenticatedBlobId(write_request.first_request_metadata());
  input.kv.key =
      std::move(*write_request.mutable_first_request_configuration());
  input.associated_metadata =
      ExtractAssociatedMetadata(write_request.first_request_metadata());
  input.protected_metadata = std::move(protected_metadata);
  ABSL_RETURN_IF_ERROR(CheckPrivacyId(input));
  inputs_.push_back(std::move(input));

  WriteFinishedResponse response;
  response.set_committed_size_bytes(data_size);
  return response;
}

absl::StatusOr<CommitResponse> GlobalReduceFn::Commit(
    CommitRequest commit_request, Context& context) {
  if (committed_) {
    return absl::FailedPreconditionError(
        "GlobalReduceFn can only be committed once.");
  }
  committed_ = true;

  CommitResponse response;
  response.mutable_stats()->set_num_inputs_committed(
      static_cast<int32_t>(inputs_.size()));
  return response;
}

absl::Status GlobalReduceFn::FinalizeReplica(google::protobuf::Any config,
                                             FnContext& context) {
  if (!committed_) {
    return absl::FailedPreconditionError(
        "Commit() must be called before Finalize().");
  }
  if (finalized_) {
    return absl::FailedPreconditionError(
        "GlobalReduceFn can only be finalized once.");
  }
  finalized_ = true;

  // Move the buffered inputs out so that they are freed once Reduce() returns.
  std::vector<Element> inputs = std::move(inputs_);
  inputs_.clear();
  ReduceContext reduce_context(context);
  ABSL_ASSIGN_OR_RETURN(ReleasableOutput output,
                        Reduce(std::move(inputs), reduce_context));
  if (!context.EmitReleasable(GetReencryptionKeyIndex(), std::move(output.kv),
                              output.src_state, output.dst_state)) {
    return absl::InternalError("Failed to emit the releasable output.");
  }
  return absl::OkStatus();
}

absl::Status GlobalReduceFn::CheckPrivacyId(const Element& input) {
  std::optional<std::string> privacy_id =
      GetPrivacyIdFromProtectedMetadata(input.protected_metadata);
  // The first input determines whether the inputs have privacy IDs, and all
  // other inputs must match it, since an input without a privacy ID could come
  // from the same privacy unit as an input with one.
  if (!has_privacy_ids_.has_value()) {
    has_privacy_ids_ = privacy_id.has_value();
  }

  if (!*has_privacy_ids_) {
    if (privacy_id.has_value()) {
      return absl::InvalidArgumentError(
          "Unexpected PrivacyIdMetadata: previous inputs have none.");
    }
    return absl::OkStatus();
  }

  if (!privacy_id.has_value()) {
    return absl::InvalidArgumentError("Missing PrivacyIdMetadata.");
  }
  if (!seen_privacy_ids_.insert(*std::move(privacy_id)).second) {
    // The privacy ID itself is sensitive, so it isn't included.
    return absl::InvalidArgumentError("Duplicate privacy id detected.");
  }
  return absl::OkStatus();
}

}  // namespace confidential_federated_compute::fns
