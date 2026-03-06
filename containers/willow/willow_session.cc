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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "containers/blob_metadata.h"
#include "containers/session.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "willow/api/server_accumulator.h"
#include "willow/proto/willow/aggregation_config.pb.h"
#include "willow/proto/willow/messages.pb.h"
#include "willow/proto/willow/server_accumulator.pb.h"
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
using ::secure_aggregation::willow::AggregationConfigProto;
using ::secure_aggregation::willow::ClientMessage;
using ::secure_aggregation::willow::ClientMessageRange;
using ::secure_aggregation::willow::FinalizedAccumulatorResult;
using ::secure_aggregation::willow::ServerAccumulator;

WillowSession::WillowSession(
    const secure_aggregation::willow::AggregationConfigProto&
        aggregation_config)
    : accumulator_(ServerAccumulator::Create(aggregation_config)) {}

absl::StatusOr<ServerAccumulator*> WillowSession::GetAccumulator() const {
  if (!accumulator_.ok()) {
    LOG(ERROR) << "Accumulator is not initialized: "
               << accumulator_.status().message();
    return accumulator_.status();
  }
  return accumulator_->get();
}

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

// Receive a single blob from the TEE host.
absl::StatusOr<WriteFinishedResponse> WillowSession::AddInput(
    std::string input) {
  ClientMessage client_message;
  if (!client_message.ParseFromString(input)) {
    LOG(ERROR) << "Blob is not a valid ClientMessage.";
    return ToWriteFinishedResponse(
        absl::InvalidArgumentError("Failed to parse client message."));
  }
  // Stash the blob for the further aggregation in a batch.
  pending_client_messages_.push_back(std::move(client_message));
  return ToWriteFinishedResponse(absl::OkStatus(), input.size());
}

// Commit the stashed batch of blobs.
absl::StatusOr<CommitResponse> WillowSession::Commit(CommitRequest request,
                                                     Context& context) {
  ClientMessageRange client_messages;
  // Add the pending client messages to the range.
  size_t num_client_messages = pending_client_messages_.size();
  for (auto& client_message : pending_client_messages_) {
    *client_messages.add_client_messages() = std::move(client_message);
  }
  pending_client_messages_.clear();

  // Unpack the range from the config.
  RangeProto range_proto;
  if (!request.configuration().UnpackTo(&range_proto)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to unpack config: ", request.configuration().DebugString()));
  }
  // Set the range start and end.
  client_messages.mutable_nonce_range()->set_start(range_proto.start());
  client_messages.mutable_nonce_range()->set_end(range_proto.end());

  FCP_ASSIGN_OR_RETURN(ServerAccumulator * accumulator, GetAccumulator());
  FCP_RETURN_IF_ERROR(
      accumulator->ProcessClientMessages(std::move(client_messages)));

  return ToCommitResponse(absl::OkStatus(), num_client_messages);
}

absl::StatusOr<WriteFinishedResponse> WillowSession::Merge(
    std::string serialized_state) {
  FCP_ASSIGN_OR_RETURN(ServerAccumulator * accumulator, GetAccumulator());
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<ServerAccumulator> other,
                       ServerAccumulator::CreateFromSerializedState(
                           std::move(serialized_state)));
  FCP_RETURN_IF_ERROR(accumulator->Merge(std::move(other)));
  return ToWriteFinishedResponse(absl::OkStatus());
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
    FCP_ASSIGN_OR_RETURN(result, Finalize());
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

absl::StatusOr<std::string> WillowSession::Compact() {
  FCP_ASSIGN_OR_RETURN(ServerAccumulator * accumulator, GetAccumulator());
  FCP_ASSIGN_OR_RETURN(std::string serialized_state,
                       accumulator->ToSerializedState());
  return serialized_state;
}

absl::StatusOr<std::string> WillowSession::Finalize() {
  FCP_ASSIGN_OR_RETURN(ServerAccumulator * accumulator, GetAccumulator());
  FCP_ASSIGN_OR_RETURN(FinalizedAccumulatorResult finalized_result,
                       std::move(*accumulator).Finalize());
  return finalized_result.SerializeAsString();
}

}  // namespace confidential_federated_compute::willow
