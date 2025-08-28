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
#include "containers/crypto.h"
#include "containers/fed_sql/private_state.h"
#include "containers/fed_sql/sensitive_columns.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/session.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "openssl/rand.h"
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
using ::confidential_federated_compute::sql::SqliteAdapter;
using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_MERGE;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::FedSqlContainerCommitConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::FinalResultConfiguration;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::WriteRequest;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Intrinsic;
using ::tensorflow_federated::aggregation::Tensor;

constexpr size_t kBlobIdSize = 16;

absl::StatusOr<std::string> CreateAssociatedData(
    absl::string_view reencryption_key,
    absl::string_view reencryption_policy_hash) {
  FCP_ASSIGN_OR_RETURN(OkpKey okp_key, OkpKey::Decode(reencryption_key));
  BlobHeader header;
  std::string blob_id(kBlobIdSize, '\0');
  (void)RAND_bytes(reinterpret_cast<unsigned char*>(blob_id.data()),
                   blob_id.size());
  header.set_blob_id(blob_id);
  header.set_key_id(okp_key.key_id);
  header.set_access_policy_sha256(std::string(reencryption_policy_hash));
  return header.SerializeAsString();
}

BlobMetadata CreateBlobMetadata(const EncryptMessageResult& encrypted_message,
                                absl::string_view associated_data) {
  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(encrypted_message.ciphertext.size());
  BlobMetadata::HpkePlusAeadMetadata* hpke_plus_aead_metadata =
      metadata.mutable_hpke_plus_aead_data();
  hpke_plus_aead_metadata->set_ciphertext_associated_data(
      std::string(associated_data));
  hpke_plus_aead_metadata->set_encrypted_symmetric_key(
      encrypted_message.encrypted_symmetric_key);
  hpke_plus_aead_metadata->set_encapsulated_public_key(
      encrypted_message.encapped_key);
  hpke_plus_aead_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header(std::string(associated_data));
  return metadata;
}
}  // namespace

KmsFedSqlSession::KmsFedSqlSession(
    std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
        aggregator,
    const std::vector<tensorflow_federated::aggregation::Intrinsic>& intrinsics,
    std::optional<SessionInferenceConfiguration> inference_configuration,
    std::optional<DpUnitParameters> dp_unit_parameters,
    absl::string_view sensitive_values_key,
    std::vector<std::string> reencryption_keys,
    absl::string_view reencryption_policy_hash,
    std::shared_ptr<PrivateState> private_state,
    std::shared_ptr<oak::crypto::SigningKeyHandle> signing_key_handle)
    : aggregator_(std::move(aggregator)),
      intrinsics_(intrinsics),
      dp_unit_parameters_(dp_unit_parameters),
      sensitive_values_key_(sensitive_values_key),
      reencryption_keys_(std::move(reencryption_keys)),
      reencryption_policy_hash_(reencryption_policy_hash),
      private_state_(std::move(private_state)),
      signing_key_handle_(signing_key_handle) {
  CHECK(reencryption_keys_.size() == 2)
      << "KmsFedSqlSession supports exactly two reencryption keys - Merge "
         "and Report.";
  CHECK_OK(confidential_federated_compute::sql::SqliteAdapter::Initialize());
  // TODO: b/427333608 - Switch to the shared model once the Gemma.cpp engine is
  // updated.
  if (inference_configuration.has_value()) {
    CHECK_OK(inference_model_.BuildModel(inference_configuration.value()));
  }
};

absl::StatusOr<std::unique_ptr<CheckpointParser>>
KmsFedSqlSession::ExecuteClientQuery(const SqlConfiguration& configuration,
                                     std::vector<Tensor> contents) {
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<SqliteAdapter> sqlite,
                       SqliteAdapter::Create());
  FCP_RETURN_IF_ERROR(sqlite->DefineTable(configuration.input_schema));
  if (!contents.empty()) {
    int num_rows = contents.at(0).num_elements();
    FCP_RETURN_IF_ERROR(
        sqlite->AddTableContents(std::move(contents), num_rows));
  }
  FCP_ASSIGN_OR_RETURN(
      std::vector<Tensor> result,
      sqlite->EvaluateQuery(configuration.query, configuration.output_columns));
  return std::make_unique<InMemoryCheckpointParser>(std::move(result));
}

absl::StatusOr<KmsFedSqlSession::EncryptedResult>
KmsFedSqlSession::EncryptIntermediateResult(absl::string_view plaintext) {
  // Use the first reencryption key for intermediate results.
  FCP_ASSIGN_OR_RETURN(
      std::string associated_data,
      CreateAssociatedData(reencryption_keys_[0], reencryption_policy_hash_));
  MessageEncryptor message_encryptor;
  FCP_ASSIGN_OR_RETURN(EncryptMessageResult encrypted_message,
                       message_encryptor.Encrypt(
                           plaintext, reencryption_keys_[0], associated_data));
  BlobMetadata metadata =
      CreateBlobMetadata(encrypted_message, associated_data);
  return EncryptedResult{.ciphertext = std::move(encrypted_message.ciphertext),
                         .metadata = std::move(metadata)};
}

absl::StatusOr<KmsFedSqlSession::EncryptedResult>
KmsFedSqlSession::EncryptFinalResult(absl::string_view plaintext) {
  // Use the second reencryption key for final results.
  FCP_ASSIGN_OR_RETURN(
      std::string associated_data,
      CreateAssociatedData(reencryption_keys_[1], reencryption_policy_hash_));
  MessageEncryptor message_encryptor;
  FCP_ASSIGN_OR_RETURN(
      EncryptMessageResult encrypted_message,
      message_encryptor.EncryptForRelease(
          plaintext, reencryption_keys_[1], associated_data,
          private_state_->initial_state,
          private_state_->budget.SerializeAsString(),
          [this](absl::string_view message) -> absl::StatusOr<std::string> {
            FCP_ASSIGN_OR_RETURN(auto signature,
                                 signing_key_handle_->Sign(message));
            return std::move(*signature.mutable_signature());
          }));

  BlobMetadata metadata =
      CreateBlobMetadata(encrypted_message, associated_data);

  FinalResultConfiguration final_result_configuration;
  final_result_configuration.set_release_token(
      std::move(encrypted_message.release_token));

  return EncryptedResult{
      .ciphertext = std::move(encrypted_message.ciphertext),
      .metadata = std::move(metadata),
      .final_result_configuration = std::move(final_result_configuration)};
}

absl::StatusOr<SessionResponse> KmsFedSqlSession::SessionWrite(
    const WriteRequest& write_request, std::string unencrypted_data) {
  FedSqlContainerWriteConfiguration write_config;
  if (!write_request.first_request_configuration().UnpackTo(&write_config)) {
    return ToSessionWriteFinishedResponse(absl::InvalidArgumentError(
        "Failed to parse FedSqlContainerWriteConfiguration."));
  }
  if (!aggregator_) {
    return absl::FailedPreconditionError("The aggregator is already released.");
  }

  switch (write_config.type()) {
    case AGGREGATION_TYPE_ACCUMULATE: {
      return SessionAccumulate(write_request.first_request_metadata(),
                               std::move(unencrypted_data));
    }
    case AGGREGATION_TYPE_MERGE: {
      return SessionMerge(write_request.first_request_metadata(),
                          std::move(unencrypted_data));
    }
    default:
      return ToSessionWriteFinishedResponse(absl::InvalidArgumentError(
          "AggCoreAggregationType must be specified."));
  }
}

template <typename T>
absl::Status PrependMessage(T message, const absl::Status& status) {
  return absl::Status(status.code(), absl::StrCat(message, status.message()));
}

absl::StatusOr<fcp::confidentialcompute::SessionResponse>
KmsFedSqlSession::SessionAccumulate(
    const fcp::confidentialcompute::BlobMetadata& metadata,
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
    return ToSessionWriteFinishedResponse(
        absl::InvalidArgumentError("Failed to parse blob header"));
  }

  // Check if there is a budget for the bucket associated with the
  // blob key.
  if (!private_state_->budget.HasRemainingBudget(blob_header.key_id())) {
    return ToSessionWriteFinishedResponse(
        absl::FailedPreconditionError("No budget remaining."));
  }

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
      parser_factory.Create(absl::Cord(std::move(unencrypted_data)));
  if (!parser.ok()) {
    return ToSessionWriteFinishedResponse(PrependMessage(
        "Failed to deserialize checkpoint for AGGREGATION_TYPE_ACCUMULATE: ",
        parser.status()));
  }
  FCP_ASSIGN_OR_RETURN(
      std::vector<Tensor> contents,
      Deserialize(sql_configuration_->input_schema, parser->get(),
                  inference_model_.GetInferenceConfiguration()));
  FCP_RETURN_IF_ERROR(HashSensitiveColumns(contents, sensitive_values_key_));
  if (inference_model_.HasModel()) {
    FCP_RETURN_IF_ERROR(inference_model_.RunInference(contents));
  }

  // Queue the blob so it can be processed on commit.
  auto blob_id = LoadBigEndian<absl::uint128>(blob_header.blob_id());
  auto [unused, inserted] = uncommitted_inputs_.emplace(
      blob_id, UncommittedInput{.contents = std::move(contents),
                                .blob_header = std::move(blob_header)});
  if (!inserted) {
    return ToSessionWriteFinishedResponse(
        absl::FailedPreconditionError("Blob rejected due to duplicate ID"));
  }

  return ToSessionWriteFinishedResponse(absl::OkStatus(),
                                        metadata.total_size_bytes());
}

absl::StatusOr<fcp::confidentialcompute::SessionResponse>
KmsFedSqlSession::SessionMerge(
    const fcp::confidentialcompute::BlobMetadata& metadata,
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
    return ToSessionWriteFinishedResponse(PrependMessage(
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
  return ToSessionWriteFinishedResponse(absl::OkStatus(),
                                        metadata.total_size_bytes());
}

absl::StatusOr<SessionResponse> KmsFedSqlSession::SessionCommit(
    const CommitRequest& commit_request) {
  // In case of an error with Accumulate, the session is terminated, since we
  // can't guarantee that the aggregator is in a valid state. If this changes,
  // consider changing this logic to no longer return an error.
  FedSqlContainerCommitConfiguration commit_config;
  if (!commit_request.configuration().UnpackTo(&commit_config)) {
    return absl::InvalidArgumentError(
        "Failed to parse FedSqlContainerCommitConfiguration.");
  }

  Interval<uint64_t> range(commit_config.range().start(),
                           commit_config.range().end());
  absl::flat_hash_set<std::string> unique_key_ids;
  std::vector<absl::Status> ignored_errors;
  for (auto& [blob_id, uncommitted_input] : uncommitted_inputs_) {
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

    if (!uncommitted_input.blob_header.key_id().empty()) {
      unique_key_ids.insert(uncommitted_input.blob_header.key_id());
    }
    std::unique_ptr<CheckpointParser> parser;
    // Execute the per-client SQL query if configured on each uncommitted blob.
    if (sql_configuration_.has_value()) {
      // TODO: b/432091990 - Update ExecuteClientQuery to accept a row
      // iterator instead of a vector of tensors.
      absl::StatusOr<std::unique_ptr<CheckpointParser>> sql_result_parser =
          ExecuteClientQuery(*sql_configuration_,
                             std::move(uncommitted_input.contents));
      if (!sql_result_parser.ok()) {
        // Ignore this blob, but continue processing other blobs.
        ignored_errors.push_back(sql_result_parser.status());
        continue;
      }
      parser = std::move(*sql_result_parser);
    } else {
      parser = std::make_unique<InMemoryCheckpointParser>(
          std::move(uncommitted_input.contents));
    }

    absl::Status accumulate_status = aggregator_->Accumulate(*parser);
    if (!accumulate_status.ok()) {
      if (absl::IsNotFound(accumulate_status)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Failed to accumulate SQL query results: ",
                         accumulate_status.message()));
      }
      return accumulate_status;
    }
  }

  for (const auto& key_id : unique_key_ids) {
    if (!range_tracker_.AddRange(key_id, commit_config.range().start(),
                                 commit_config.range().end())) {
      return absl::FailedPreconditionError(
          "Failed to commit due to conflicting ranges");
    }
  }

  SessionResponse response =
      ToSessionCommitResponse(absl::OkStatus(), uncommitted_inputs_.size(), {});
  uncommitted_inputs_.clear();
  return response;
}

// Runs the requested finalization operation and write the uncompressed result
// to the stream. After finalization, the session state is no longer mutable.
//
// Any errors in HandleFinalize kill the stream, since the stream can no
// longer be modified after the Finalize call.
absl::StatusOr<SessionResponse> KmsFedSqlSession::FinalizeSession(
    const FinalizeRequest& request, const BlobMetadata& unused) {
  FedSqlContainerFinalizeConfiguration finalize_config;
  if (!request.configuration().UnpackTo(&finalize_config)) {
    return absl::InvalidArgumentError(
        "Failed to parse FedSqlContainerFinalizeConfiguration.");
  }
  if (!aggregator_) {
    return absl::FailedPreconditionError("The aggregator is already released.");
  }

  std::string result;
  BlobMetadata result_metadata;
  std::optional<FinalResultConfiguration> final_result_configuration;
  switch (finalize_config.type()) {
    case fcp::confidentialcompute::FINALIZATION_TYPE_REPORT: {
      // Update the private state
      FCP_RETURN_IF_ERROR(private_state_->budget.UpdateBudget(range_tracker_));

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
      // Using the scope below ensures that both CheckpointBuilder and Cord
      // are promptly deleted.
      absl::Cord checkpoint_cord;
      {
        FederatedComputeCheckpointBuilderFactory builder_factory;
        std::unique_ptr<CheckpointBuilder> checkpoint_builder =
            builder_factory.Create();
        FCP_RETURN_IF_ERROR(aggregator_->Report(*checkpoint_builder));
        aggregator_.reset();
        FCP_ASSIGN_OR_RETURN(checkpoint_cord, checkpoint_builder->Build());
      }

      // Encrypt the final result.
      FCP_ASSIGN_OR_RETURN(auto encrypted_result,
                           EncryptFinalResult(checkpoint_cord.Flatten()));
      result_metadata = std::move(encrypted_result.metadata);
      result = std::move(encrypted_result.ciphertext);
      final_result_configuration =
          std::move(encrypted_result.final_result_configuration);
      break;
    }
    case FINALIZATION_TYPE_SERIALIZE: {
      // Serialize the aggregator and bundle it with the range tracker.
      FCP_ASSIGN_OR_RETURN(std::string serialized_data,
                           std::move(*aggregator_).Serialize());
      aggregator_.reset();
      serialized_data = BundleRangeTracker(serialized_data, range_tracker_);
      // Encrypt the bundled blob.
      FCP_ASSIGN_OR_RETURN(auto encrypted_result,
                           EncryptIntermediateResult(serialized_data));
      result_metadata = std::move(encrypted_result.metadata);
      result = std::move(encrypted_result.ciphertext);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Finalize configuration must specify the finalization type.");
  }

  SessionResponse response;
  ReadResponse* read_response = response.mutable_read();
  read_response->set_finish_read(true);
  *(read_response->mutable_data()) = std::move(result);
  *(read_response->mutable_first_response_metadata()) = result_metadata;
  if (final_result_configuration.has_value()) {
    google::protobuf::Any config;
    config.PackFrom(std::move(final_result_configuration.value()));
    *(read_response->mutable_first_response_configuration()) =
        std::move(config);
  }
  return response;
}

absl::Status KmsFedSqlSession::ConfigureSession(
    fcp::confidentialcompute::SessionRequest configure_request) {
  if (!configure_request.configure().has_configuration()) {
    return absl::InvalidArgumentError(
        "`configure` must be set on SessionRequest.");
  }
  SqlQuery sql_query;
  if (!configure_request.configure().configuration().UnpackTo(&sql_query)) {
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

  return absl::OkStatus();
}
}  // namespace confidential_federated_compute::fed_sql
