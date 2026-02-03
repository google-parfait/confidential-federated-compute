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
#include "containers/fed_sql/kms_session.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "containers/big_endian.h"
#include "containers/fed_sql/private_state.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/session.h"
#include "containers/sql/input.h"
#include "containers/sql/row_set.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/time_window_utilities.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "google/protobuf/message.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::fed_sql {

namespace {
using ::confidential_federated_compute::sql::Input;
using ::confidential_federated_compute::sql::RowLocation;
using ::confidential_federated_compute::sql::RowSet;
using ::confidential_federated_compute::sql::SqliteAdapter;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivateLoggerEntryKey;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE_PRIVATE_STATE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_MERGE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_MERGE_PRIVATE_STATE;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::FedSqlContainerCommitConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerPartitionedOutputConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_PARTITION;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT_PARTITION;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT_PRIVATE_STATE;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE_PRIVATE_STATE;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::FinalizeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Message;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Intrinsic;
using ::tensorflow_federated::aggregation::Tensor;

absl::StatusOr<Input> CreateInputFromMessageCheckpoint(
    BlobHeader blob_header, CheckpointParser* checkpoint,
    MessageFactory& message_factory, absl::string_view on_device_query_name) {
  std::string column_prefix = absl::StrCat(on_device_query_name, "/");
  FCP_ASSIGN_OR_RETURN(Tensor entry_tensor,
                       checkpoint->GetTensor(absl::StrCat(
                           column_prefix, kPrivateLoggerEntryKey)));
  if (entry_tensor.dtype() !=
      tensorflow_federated::aggregation::DataType::DT_STRING) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must be a string tensor", kPrivateLoggerEntryKey));
  }
  FCP_ASSIGN_OR_RETURN(
      Tensor time_tensor,
      checkpoint->GetTensor(absl::StrCat(column_prefix, kEventTimeColumnName)));
  if (time_tensor.dtype() !=
      tensorflow_federated::aggregation::DataType::DT_STRING) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`%s` tensor must be a string tensor", kEventTimeColumnName));
  }

  // Rename the time tensor to remove the column prefix. Pipelines that process
  // Message-based checkpoints don't use the column name prefix.
  FCP_RETURN_IF_ERROR(time_tensor.set_name(kEventTimeColumnName));

  std::vector<std::unique_ptr<google::protobuf::Message>> messages;
  messages.reserve(entry_tensor.num_elements());
  for (const absl::string_view entry :
       entry_tensor.AsSpan<absl::string_view>()) {
    std::unique_ptr<google::protobuf::Message> message(
        message_factory.NewMessage());
    if (!message->ParseFromArray(entry.data(), entry.size())) {
      return absl::InvalidArgumentError("Failed to parse proto");
    }
    messages.push_back(std::move(message));
  }

  std::vector<Tensor> system_columns;
  system_columns.reserve(1);
  system_columns.push_back(std::move(time_tensor));
  return Input::CreateFromMessages(std::move(messages),
                                   std::move(system_columns), blob_header);
}

}  // namespace

KmsFedSqlSession::KmsFedSqlSession(
    std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
        aggregator,
    const std::vector<tensorflow_federated::aggregation::Intrinsic>& intrinsics,
    std::optional<SessionInferenceConfiguration> inference_configuration,
    std::optional<DpUnitParameters> dp_unit_parameters,
    std::shared_ptr<PrivateState> private_state,
    const absl::flat_hash_set<std::string>& expired_key_ids,
    std::shared_ptr<MessageFactory> message_factory,
    absl::string_view on_device_query_name,
    confidential_federated_compute::Decryptor& decryptor)
    : aggregator_(std::move(aggregator)),
      intrinsics_(intrinsics),
      private_state_(std::move(private_state)),
      dp_unit_parameters_(dp_unit_parameters),
      message_factory_(std::move(message_factory)),
      on_device_query_name_(on_device_query_name),
      decryptor_(decryptor) {
  CHECK_OK(confidential_federated_compute::sql::SqliteAdapter::Initialize());
  range_tracker_.SetExpiredKeys(expired_key_ids);
  // TODO: b/427333608 - Switch to the shared model once the Gemma.cpp engine is
  // updated.
  if (inference_configuration.has_value()) {
    CHECK_OK(inference_model_.BuildModel(inference_configuration.value()));
  }
};

absl::StatusOr<WriteFinishedResponse> KmsFedSqlSession::Write(
    WriteRequest request, std::string unencrypted_data, Context& context) {
  FedSqlContainerWriteConfiguration write_config;
  if (!request.first_request_configuration().UnpackTo(&write_config)) {
    return ToWriteFinishedResponse(absl::InvalidArgumentError(
        "Failed to parse FedSqlContainerWriteConfiguration."));
  }
  if (!aggregator_) {
    return absl::FailedPreconditionError("The aggregator is already released.");
  }

  switch (write_config.type()) {
    case AGGREGATION_TYPE_ACCUMULATE: {
      return Accumulate(std::move(*request.mutable_first_request_metadata()),
                        std::move(unencrypted_data));
    }
    case AGGREGATION_TYPE_MERGE: {
      return Merge(std::move(*request.mutable_first_request_metadata()),
                   std::move(unencrypted_data));
    }
    case AGGREGATION_TYPE_ACCUMULATE_PRIVATE_STATE: {
      return AccumulatePrivateState(
          request.first_request_metadata().total_size_bytes(),
          std::move(unencrypted_data));
    }
    case AGGREGATION_TYPE_MERGE_PRIVATE_STATE:
      return MergePrivateState(
          request.first_request_metadata().total_size_bytes(),
          std::move(unencrypted_data));
    default:
      return ToWriteFinishedResponse(absl::InvalidArgumentError(
          "AggCoreAggregationType must be specified."));
  }
}

template <typename T>
absl::Status PrependMessage(T message, const absl::Status& status) {
  return absl::Status(status.code(), absl::StrCat(message, status.message()));
}

absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse>
KmsFedSqlSession::Accumulate(fcp::confidentialcompute::BlobMetadata metadata,
                             std::string unencrypted_data) {
  // TODO: Use
  // https://github.com/google-parfait/federated-compute/blob/main/fcp/base/scheduler.h
  // to asynchronously handle deserializing the checkpoint when it is
  // initially written to the session.
  BlobHeader blob_header;
  // The metadata is expected to has encryption with KMS.
  if (!metadata.has_hpke_plus_aead_data() ||
      !metadata.hpke_plus_aead_data().has_kms_symmetric_key_associated_data()) {
    // Not having this indicates a problem with the configuration, which means
    // the session should be aborted.
    return absl::InternalError(
        "Unexpected blob metadata: must have encryption with KMS.");
  }

  if (!blob_header.ParseFromString(metadata.hpke_plus_aead_data()
                                       .kms_symmetric_key_associated_data()
                                       .record_header())) {
    return ToWriteFinishedResponse(
        absl::InvalidArgumentError("Failed to parse blob header"));
  }

  // Check if there is a budget for the bucket associated with the
  // blob key.
  auto blob_id = LoadBigEndian<absl::uint128>(blob_header.blob_id());
  uint64_t blob_id_high64 = absl::Uint128High64(blob_id);
  if (!private_state_->budget.HasRemainingBudget(blob_header.key_id(),
                                                 blob_id_high64)) {
    return ToWriteFinishedResponse(
        absl::FailedPreconditionError("No budget remaining."));
  }

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
      parser_factory.Create(absl::Cord(std::move(unencrypted_data)));
  if (!parser.ok()) {
    return ToWriteFinishedResponse(PrependMessage(
        "Failed to deserialize checkpoint for AGGREGATION_TYPE_ACCUMULATE: ",
        parser.status()));
  }
  absl::StatusOr<Input> input;
  if (message_factory_ != nullptr) {
    input = CreateInputFromMessageCheckpoint(
        blob_header, parser->get(), *message_factory_, on_device_query_name_);
    // TODO: handle sensitive columns for Message checkpoints.
  } else {
    absl::StatusOr<std::vector<Tensor>> contents =
        Deserialize(sql_configuration_->input_schema, parser->get(),
                    inference_model_.GetInferenceConfiguration());
    if (!contents.ok()) {
      return ToWriteFinishedResponse(
          PrependMessage("Failed to deserialize checkpoint for "
                         "AGGREGATION_TYPE_ACCUMULATE: ",
                         contents.status()));
    }
    input = Input::CreateFromTensors(std::move(contents.value()), blob_header);
  }
  if (!input.ok()) {
    return ToWriteFinishedResponse(PrependMessage(
        "Failed to create input for AGGREGATION_TYPE_ACCUMULATE: ",
        input.status()));
  }
  if (inference_model_.HasModel()) {
    FCP_RETURN_IF_ERROR(inference_model_.RunInference(*input));
  }

  // TODO: Calculate the DP unit for each row in the blob and add a RowLocation
  // to uncommitted_row_locations_ to track it.

  auto [unused, inserted] = uncommitted_blob_ids_.insert(blob_id);
  if (!inserted) {
    return ToWriteFinishedResponse(
        absl::FailedPreconditionError("Blob rejected due to duplicate ID"));
  }

  uncommitted_inputs_.push_back(*std::move(input));

  return ToWriteFinishedResponse(absl::OkStatus(), metadata.total_size_bytes());
}

absl::StatusOr<WriteFinishedResponse> KmsFedSqlSession::AccumulatePrivateState(
    int64_t total_size_bytes, std::string release_token) {
  absl::StatusOr<fcp::confidential_compute::UnwrappedReleaseToken>
      unwrapped_release_token = decryptor_.UnwrapReleaseToken(release_token);
  if (!unwrapped_release_token.ok()) {
    return ToWriteFinishedResponse(
        PrependMessage("Failed to unwrap release token in "
                       "AGGREGATION_TYPE_ACCUMULATE_PRIVATE_STATE: ",
                       unwrapped_release_token.status()));
  }

  if (unwrapped_release_token->dst_state == std::nullopt) {
    return ToWriteFinishedResponse(absl::InvalidArgumentError(
        "Corrupt release token in AGGREGATION_TYPE_ACCUMULATE_PRIVATE_STATE. "
        "dst_state should be non-empty, but found empty dst_state."));
  }

  absl::StatusOr<RangeTracker> range_tracker =
      RangeTracker::Parse(unwrapped_release_token->dst_state.value());
  if (!range_tracker.ok()) {
    return ToWriteFinishedResponse(
        PrependMessage("Failed to parse range tracker for dst_state in "
                       "AGGREGATION_TYPE_ACCUMULATE_PRIVATE_STATE: ",
                       range_tracker.status()));
  }

  if (!partition_private_state_.AddPartition(
          *range_tracker, unwrapped_release_token->serialized_symmetric_key)) {
    return ToWriteFinishedResponse(absl::FailedPreconditionError(
        "Failed to add partition's private state due to conflicts."));
  }

  return ToWriteFinishedResponse(absl::OkStatus(), total_size_bytes);
}

absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse>
KmsFedSqlSession::Merge(fcp::confidentialcompute::BlobMetadata metadata,
                        std::string unencrypted_data) {
  //  Merges can be immediately processed.
  FCP_ASSIGN_OR_RETURN(RangeTracker other_range_tracker,
                       UnbundleRangeTracker(unencrypted_data));
  if (!range_tracker_.Merge(other_range_tracker)) {
    return absl::FailedPreconditionError(
        "Failed to merge due to conflicting ranges.");
  }

  absl::StatusOr<std::unique_ptr<CheckpointAggregator>> other =
      CheckpointAggregator::Deserialize(&intrinsics_,
                                        std::move(unencrypted_data));
  if (!other.ok()) {
    return ToWriteFinishedResponse(PrependMessage(
        "Failed to deserialize checkpoint for AGGREGATION_TYPE_MERGE: ",
        other.status()));
  }
  // In case of an error with MergeWith, the session is terminated, since we
  // can't guarantee that the aggregator is in a valid state. If this changes,
  // consider changing this logic to no longer return an error.
  absl::Status merge_status = aggregator_->MergeWith(std::move(*other.value()));
  if (!merge_status.ok()) {
    LOG(ERROR) << "CheckpointAggregator::MergeWith failed: " << merge_status;
    return merge_status;
  }
  return ToWriteFinishedResponse(absl::OkStatus(), metadata.total_size_bytes());
}

absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse>
KmsFedSqlSession::MergePrivateState(int64_t total_size_bytes,
                                    std::string unencrypted_data) {
  absl::StatusOr<PartitionPrivateState> partition_private_state =
      PartitionPrivateState::Parse(unencrypted_data);
  if (!partition_private_state.ok()) {
    return ToWriteFinishedResponse(
        PrependMessage("Failed to parse PartitionPrivateState in "
                       "AGGREGATION_TYPE_MERGE_PRIVATE_STATE: ",
                       partition_private_state.status()));
  }
  if (!partition_private_state_.Merge(partition_private_state.value())) {
    return ToWriteFinishedResponse(absl::InvalidArgumentError(
        "Failed to merge PartitionPrivateState due to conflicts."));
  }
  return ToWriteFinishedResponse(absl::OkStatus(), total_size_bytes);
}

absl::StatusOr<std::vector<absl::Status>>
KmsFedSqlSession::CommitRowsGroupingByInput(
    std::vector<Input>&& uncommitted_inputs, const Interval<uint64_t>& range) {
  std::vector<absl::Status> ignored_errors;
  // Iterate over uncommitted_blob_ids_ and check that they're in the specified
  // range.
  for (auto blob_id : uncommitted_blob_ids_) {
    // Use the high 64 bit of the blob_id to check whether the blob is
    // in the specified range.
    uint64_t blob_id_high64 = absl::Uint128High64(blob_id);
    if (!range.Contains(blob_id_high64)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Failed to commit due to blob ID conflicting with the "
                       "range. Range: [",
                       range.start(), ", ", range.end(),
                       "), blob id (high 8 bytes): ", blob_id_high64));
    }
  }

  for (auto& uncommitted_input : uncommitted_inputs) {
    std::unique_ptr<CheckpointParser> parser;
    // Execute the per-client SQL query if configured on each uncommitted blob.
    if (sql_configuration_.has_value()) {
      absl::Span<Input> storage = absl::MakeSpan(&uncommitted_input, 1);
      std::vector<RowLocation> row_locations =
          CreateRowLocationsForAllRows(uncommitted_input.GetRowCount());
      FCP_ASSIGN_OR_RETURN(RowSet row_set,
                           RowSet::Create(row_locations, storage));
      absl::StatusOr<std::vector<Tensor>> sql_result =
          ExecuteClientQuery(*sql_configuration_, row_set);
      if (!sql_result.ok()) {
        // Ignore this blob, but continue processing other blobs.
        ignored_errors.push_back(sql_result.status());
        continue;
      }
      parser =
          std::make_unique<InMemoryCheckpointParser>(*std::move(sql_result));
    } else {
      absl::StatusOr<std::vector<Tensor>> tensors =
          std::move(uncommitted_input).MoveToTensors();
      if (!tensors.ok()) {
        // Ignore this blob, but continue processing other blobs.
        ignored_errors.push_back(tensors.status());
        continue;
      }
      parser = std::make_unique<InMemoryCheckpointParser>(*std::move(tensors));
    }

    absl::Status accumulate_status = aggregator_->Accumulate(*parser);
    // Use the "Invalid Argument" error code to detect bad inputs
    if (accumulate_status.code() == absl::StatusCode::kInvalidArgument) {
      LOG(INFO) << "Invalid input skipped";
      ignored_errors.push_back(accumulate_status);
      continue;
    }
    // For other bad codes, the session is terminated since we can't guarantee
    // that the aggregator is in a valid state.
    if (!accumulate_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to accumulate SQL query results: ",
                       accumulate_status.message()));
    }
  }
  return ignored_errors;
}

absl::StatusOr<CommitResponse> KmsFedSqlSession::Commit(
    CommitRequest commit_request, Context& context) {
  FedSqlContainerCommitConfiguration commit_config;
  if (!commit_request.configuration().UnpackTo(&commit_config)) {
    return absl::InvalidArgumentError(
        "Failed to parse FedSqlContainerCommitConfiguration.");
  }

  Interval<uint64_t> range(commit_config.range().start(),
                           commit_config.range().end());
  absl::flat_hash_set<std::string> unique_key_ids;
  for (auto& uncommitted_input : uncommitted_inputs_) {
    // TODO: Once we switch to using DP time unit for budget buckets, we'll
    // need to use DP time units here instead of key ID.
    if (!uncommitted_input.GetBlobHeader().key_id().empty()) {
      unique_key_ids.insert(uncommitted_input.GetBlobHeader().key_id());
    }
  }

  int num_committed = uncommitted_inputs_.size();
  // TODO: Commit rows by DP unit if DP parameters are configured.
  FCP_ASSIGN_OR_RETURN(
      std::vector<absl::Status> ignored_errors,
      CommitRowsGroupingByInput(std::move(uncommitted_inputs_), range));

  for (const auto& key_id : unique_key_ids) {
    if (!range_tracker_.AddRange(key_id, commit_config.range().start(),
                                 commit_config.range().end())) {
      return absl::FailedPreconditionError(
          "Failed to commit due to conflicting ranges");
    }
  }

  uncommitted_inputs_.clear();
  uncommitted_blob_ids_.clear();
  return ToCommitResponse(absl::OkStatus(), num_committed);
}

// Runs the requested finalization operation and write the uncompressed result
// to the stream. After finalization, the session state is no longer mutable.
//
// Any errors in HandleFinalize kill the stream, since the stream can no
// longer be modified after the Finalize call.
absl::StatusOr<FinalizeResponse> KmsFedSqlSession::Finalize(
    FinalizeRequest request, BlobMetadata input_metadata, Context& context) {
  FedSqlContainerFinalizeConfiguration finalize_config;
  if (!request.configuration().UnpackTo(&finalize_config)) {
    return absl::InvalidArgumentError(
        "Failed to parse FedSqlContainerFinalizeConfiguration.");
  }
  if (!aggregator_) {
    return absl::FailedPreconditionError("The aggregator is already released.");
  }

  switch (finalize_config.type()) {
    case fcp::confidentialcompute::FINALIZATION_TYPE_REPORT: {
      return Report(context);
    }
    case FINALIZATION_TYPE_SERIALIZE: {
      return Serialize(context);
    }
    case FINALIZATION_TYPE_PARTITION: {
      return Partition(context, finalize_config.num_partitions());
    }
    case FINALIZATION_TYPE_REPORT_PARTITION: {
      return ReportPartition(context);
    }
    case FINALIZATION_TYPE_SERIALIZE_PRIVATE_STATE: {
      return SerializePrivateState(context);
    }
    case FINALIZATION_TYPE_REPORT_PRIVATE_STATE: {
      return ReportPrivateState(context);
    }
    default:
      return absl::InvalidArgumentError(
          "Finalize configuration must specify the finalization type.");
  }
}

absl::StatusOr<fcp::confidentialcompute::FinalizeResponse>
KmsFedSqlSession::Serialize(Context& context) {
  // Serialize the aggregator and bundle it with the range tracker.
  FCP_ASSIGN_OR_RETURN(std::string serialized_data,
                       std::move(*aggregator_).Serialize());
  aggregator_.reset();
  std::string bundled_data =
      BundleRangeTracker(std::move(serialized_data), range_tracker_);
  if (!context.EmitEncrypted(/* reencryption_key_index*/ 0,
                             std::move(bundled_data))) {
    return absl::InternalError("Failed to emit encrypted result.");
  }

  return FinalizeResponse{};
}

absl::StatusOr<fcp::confidentialcompute::FinalizeResponse>
KmsFedSqlSession::SerializePrivateState(Context& context) {
  std::string serialized_private_state =
      partition_private_state_.SerializeAsString();
  if (!context.EmitEncrypted(/* reencryption_key_index*/ 1,
                             std::move(serialized_private_state))) {
    return absl::InternalError(
        "Failed to emit encrypted serialized private state.");
  }

  return FinalizeResponse{};
}

absl::StatusOr<fcp::confidentialcompute::FinalizeResponse>
KmsFedSqlSession::Partition(Context& context, uint64_t num_partitions) {
  if (num_partitions == 0) {
    return absl::InvalidArgumentError(
        "Number of partitions must be greater than zero.");
  }

  FCP_ASSIGN_OR_RETURN(std::vector<std::string> partitions,
                       std::move(*aggregator_).Partition(num_partitions));
  aggregator_.reset();
  CHECK(partitions.size() == num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    range_tracker_.SetPartitionIndex(i);
    std::string bundled_partition =
        BundleRangeTracker(std::move(partitions[i]), range_tracker_);
    FedSqlContainerPartitionedOutputConfiguration partition_config;
    partition_config.set_partition_index(i);
    ::google::protobuf::Any config;
    config.PackFrom(partition_config);
    if (!context.EmitEncrypted(
            /* reencryption_key_index*/ 0,
            KV(std::move(config), std::move(bundled_partition)))) {
      return absl::InternalError("Failed to emit encrypted partition result.");
    }
  }

  return FinalizeResponse{};
}

absl::StatusOr<fcp::confidentialcompute::FinalizeResponse>
KmsFedSqlSession::Report(Context& context) {
  // Update the private state
  FCP_RETURN_IF_ERROR(private_state_->budget.UpdateBudget(range_tracker_));

  // Produce the final report.
  FCP_ASSIGN_OR_RETURN(absl::Cord checkpoint, BuildReport());
  // Emit the final encrypted result.
  std::string release_token;
  if (!context.EmitReleasable(
          /* reencryption_key_index*/ 1, std::string(checkpoint.Flatten()),
          private_state_->initial_state,
          private_state_->budget.SerializeAsString(), release_token)) {
    return absl::InternalError("Failed to emit releasable final result.");
  }

  FinalizeResponse response;
  *response.mutable_release_token() = std::move(release_token);
  return response;
}

absl::StatusOr<fcp::confidentialcompute::FinalizeResponse>
KmsFedSqlSession::ReportPartition(Context& context) {
  // Produce the final report.
  FCP_ASSIGN_OR_RETURN(absl::Cord checkpoint, BuildReport());
  // If there is unlimited budget, emit an unencrypted result
  // since budget tracking is not required.
  if (private_state_->budget.HasUnlimitedBudget()) {
    if (!context.EmitUnencrypted(std::string(checkpoint.Flatten()))) {
      return absl::InternalError(
          "Failed to emit unencrypted result"
          " for the partition.");
    }
    return FinalizeResponse{};
  }

  // Otherwise, encrypt the result and return a release token for
  // the partition. This release token will be unwrapped by the next
  // stage of the worker when merging the range tracker state.
  std::string release_token;
  if (!context.EmitReleasable(
          /* reencryption_key_index*/ 1, std::string(checkpoint.Flatten()),
          std::nullopt, range_tracker_.SerializeAsString(), release_token)) {
    return absl::InternalError("Failed to emit releasable partition result.");
  }
  FinalizeResponse response;
  *response.mutable_release_token() = std::move(release_token);
  return response;
}

absl::StatusOr<fcp::confidentialcompute::FinalizeResponse>
KmsFedSqlSession::ReportPrivateState(Context& context) {
  // Update the private state
  FCP_RETURN_IF_ERROR(private_state_->budget.UpdateBudget(
      partition_private_state_.GetPerKeyRanges(),
      partition_private_state_.GetExpiredKeys()));
  // Get all the symmetric keys for the partitions.
  std::string serialized_keys = partition_private_state_.GetSerializedKeys();
  // Emit the final encrypted result.
  std::string release_token;
  if (!context.EmitReleasable(
          /* reencryption_key_index*/ 2, serialized_keys,
          private_state_->initial_state,
          private_state_->budget.SerializeAsString(), release_token)) {
    return absl::InternalError("Failed to emit releasable final result.");
  }

  FinalizeResponse response;
  *response.mutable_release_token() = std::move(release_token);
  return response;
}

absl::StatusOr<absl::Cord> KmsFedSqlSession::BuildReport() {
  if (!aggregator_->CanReport()) {
    return absl::FailedPreconditionError(
        "The aggregation can't be completed due to failed "
        "preconditions.");
  }
  // Fail if there were no valid inputs, as this likely indicates some
  // issue with configuration of the overall workload.
  FCP_ASSIGN_OR_RETURN(int num_checkpoints_aggregated,
                       aggregator_->GetNumCheckpointsAggregated());
  if (num_checkpoints_aggregated < 1) {
    return absl::InvalidArgumentError(
        "The aggregation can't be successfully completed because no "
        "inputs were aggregated.\n"
        "This may be because inputs were ignored due to an earlier "
        "error.");
  }

  // Extract unencrypted checkpoint from the aggregator.
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  FCP_RETURN_IF_ERROR(aggregator_->Report(*checkpoint_builder));
  aggregator_.reset();
  FCP_ASSIGN_OR_RETURN(absl::Cord checkpoint, checkpoint_builder->Build());
  return checkpoint;
}

absl::StatusOr<ConfigureResponse> KmsFedSqlSession::Configure(
    ConfigureRequest request, Context& context) {
  SqlQuery sql_query;
  if (!request.configuration().UnpackTo(&sql_query)) {
    return absl::InvalidArgumentError("SQL configuration cannot be unpacked.");
  }
  if (sql_query.database_schema().table_size() != 1) {
    return absl::InvalidArgumentError(
        "SQL query input or output schema does not contain exactly "
        "one table schema.");
  }
  if (sql_query.database_schema().table(0).column_size() == 0) {
    return absl::InvalidArgumentError("SQL query input schema has no columns.");
  }
  if (sql_configuration_ != std::nullopt) {
    return absl::FailedPreconditionError(
        "Session can only be configured once.");
  }

  sql_configuration_.emplace(
      SqlConfiguration{std::move(sql_query.raw_sql()),
                       std::move(sql_query.database_schema().table(0)),
                       std::move(sql_query.output_columns())});

  return ConfigureResponse{};
}
}  // namespace confidential_federated_compute::fed_sql
