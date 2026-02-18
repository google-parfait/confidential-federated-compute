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

#include "willow_session.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "containers/blob_metadata.h"
#include "containers/session.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "willow_op.pb.h"

namespace confidential_federated_compute::willow {

using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::FinalizeResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;

absl::StatusOr<ConfigureResponse> WillowSession::Configure(
    ConfigureRequest request, Context& context) {
  return ConfigureResponse{};
}

absl::StatusOr<WriteFinishedResponse> WillowSession::Write(
    WriteRequest request, std::string unencrypted_data, Context& context) {
  WillowOp op;
  if (!request.first_request_configuration().UnpackTo(&op)) {
    return ToWriteFinishedResponse(absl::InvalidArgumentError(
        "WillowSession::Write: failed to parse Op."));
  }
  if (op.kind() == WillowOp::ADD_INPUT) {
    return AddInput(std::move(unencrypted_data));
  } else if (op.kind() == WillowOp::MERGE) {
    return Merge(std::move(unencrypted_data));
  } else {
    return ToWriteFinishedResponse(absl::InvalidArgumentError(absl::StrCat(
        "SessWillowSessionion::Write: unexpected op: ", op.DebugString())));
  }
}

absl::StatusOr<CommitResponse> WillowSession::Commit(CommitRequest request,
                                                     Context& context) {
  return ToCommitResponse(absl::OkStatus());
}

absl::StatusOr<FinalizeResponse> WillowSession::Finalize(
    FinalizeRequest request, BlobMetadata input_metadata, Context& context) {
  WillowOp op;
  if (!request.configuration().UnpackTo(&op)) {
    return absl::InvalidArgumentError(
        "WillowSession::Finalize: failed to parse Op.");
  }
  std::string result;
  if (op.kind() == WillowOp::COMPACT) {
    FCP_ASSIGN_OR_RETURN(result, Compact());
  } else if (op.kind() == WillowOp::FINALIZE) {
    FCP_ASSIGN_OR_RETURN(result, Compact());
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "WillowSession::Finalize: unexpected op ", op.DebugString()));
  }
  if (!context.EmitUnencrypted(std::move(result))) {
    return absl::InternalError(
        "WillowSession::Finalize: failed to emit result.");
  }
  return FinalizeResponse{};
}

absl::StatusOr<WriteFinishedResponse> WillowSession::AddInput(
    std::string input) {
  return WriteFinishedResponse{};
}
absl::StatusOr<WriteFinishedResponse> WillowSession::Merge(std::string input) {
  return WriteFinishedResponse{};
}
absl::StatusOr<std::string> WillowSession::Compact() { return ""; }
absl::StatusOr<std::string> WillowSession::Finalize() { return ""; }

}  // namespace confidential_federated_compute::willow
