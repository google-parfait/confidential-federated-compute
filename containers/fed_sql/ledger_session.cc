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
#include "containers/fed_sql/ledger_session.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "containers/fed_sql/sensitive_columns.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/session.h"
#include "containers/sql/row_set.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::confidential_federated_compute::sql::SqliteAdapter;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_MERGE;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::WriteRequest;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::DT_DOUBLE;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Intrinsic;
using ::tensorflow_federated::aggregation::kDeltaIndex;
using ::tensorflow_federated::aggregation::kEpsilonIndex;
using ::tensorflow_federated::aggregation::Tensor;

}  // namespace

FedSqlSession::FedSqlSession(
    std::unique_ptr<tensorflow_federated::aggregation::CheckpointAggregator>
        aggregator,
    const std::vector<tensorflow_federated::aggregation::Intrinsic>& intrinsics,
    std::optional<SessionInferenceConfiguration> inference_configuration_,
    const std::optional<uint32_t> serialize_output_access_policy_node_id,
    const std::optional<uint32_t> report_output_access_policy_node_id,
    absl::string_view sensitive_values_key)
    : aggregator_(std::move(aggregator)),
      intrinsics_(intrinsics),
      serialize_output_access_policy_node_id_(
          serialize_output_access_policy_node_id),
      report_output_access_policy_node_id_(report_output_access_policy_node_id),
      sensitive_values_key_(sensitive_values_key) {
  if (inference_configuration_.has_value()) {
    CHECK_OK(inference_model_.BuildModel(inference_configuration_.value()));
  }
};

absl::StatusOr<std::unique_ptr<CheckpointParser>>
FedSqlSession::ExecuteInferenceAndClientQuery(
    const SqlConfiguration& configuration, CheckpointParser* parser) {
  FCP_ASSIGN_OR_RETURN(
      std::vector<Tensor> contents,
      Deserialize(configuration.input_schema, parser,
                  inference_model_.GetInferenceConfiguration()));
  FCP_RETURN_IF_ERROR(HashSensitiveColumns(contents, sensitive_values_key_));
  FCP_ASSIGN_OR_RETURN(sql::Input input,
                       sql::Input::CreateFromTensors(std::move(contents), {}));
  if (inference_model_.HasModel()) {
    FCP_RETURN_IF_ERROR(inference_model_.RunInference(input));
  }
  absl::Span<sql::Input> storage = absl::MakeSpan(&input, 1);
  std::vector<sql::RowLocation> row_locations =
      CreateRowLocationsForAllRows(input.GetRowCount());
  FCP_ASSIGN_OR_RETURN(sql::RowSet row_set,
                       sql::RowSet::Create(row_locations, storage));
  FCP_ASSIGN_OR_RETURN(std::vector<Tensor> sql_result,
                       ExecuteClientQuery(configuration, row_set));
  return std::make_unique<InMemoryCheckpointParser>(std::move(sql_result));
}

absl::StatusOr<SessionResponse> FedSqlSession::SessionWrite(
    const WriteRequest& write_request, std::string unencrypted_data) {
  FedSqlContainerWriteConfiguration write_config;
  if (!write_request.first_request_configuration().UnpackTo(&write_config)) {
    return ToSessionWriteFinishedResponse(absl::InvalidArgumentError(
        "Failed to parse FedSqlContainerWriteConfiguration."));
  }
  if (!aggregator_) {
    return absl::FailedPreconditionError("The aggregator is already released.");
  }

  // In case of an error with Accumulate or MergeWith, the session is
  // terminated, since we can't guarantee that the aggregator is in a valid
  // state. If this changes, consider changing this logic to no longer return an
  // error.
  switch (write_config.type()) {
    case AGGREGATION_TYPE_ACCUMULATE: {
      FederatedComputeCheckpointParserFactory parser_factory;
      absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
          parser_factory.Create(absl::Cord(std::move(unencrypted_data)));
      if (!parser.ok()) {
        return ToSessionWriteFinishedResponse(
            absl::Status(parser.status().code(),
                         absl::StrCat("Failed to deserialize checkpoint for "
                                      "AGGREGATION_TYPE_ACCUMULATE: ",
                                      parser.status().message())));
      }
      if (sql_configuration_ != std::nullopt) {
        absl::StatusOr<std::unique_ptr<CheckpointParser>> sql_result_parser =
            ExecuteInferenceAndClientQuery(*sql_configuration_, parser->get());
        if (!sql_result_parser.ok()) {
          return ToSessionWriteFinishedResponse(
              absl::Status(sql_result_parser.status().code(),
                           absl::StrCat("Failed to execute SQL query: ",
                                        sql_result_parser.status().message())));
        }
        parser = std::move(sql_result_parser);
      }
      absl::Status accumulate_status = aggregator_->Accumulate(*parser.value());
      if (!accumulate_status.ok()) {
        if (absl::IsNotFound(accumulate_status)) {
          return absl::InvalidArgumentError(
              absl::StrCat("Failed to accumulate SQL query results: ",
                           accumulate_status.message()));
        }
        return accumulate_status;
      }
      break;
    }
    case AGGREGATION_TYPE_MERGE: {
      // TODO: Avoid copying unencrypted_cord back string, which can be
      // achieved by passing a cord to CheckpointAggregator implementing parsing
      // from Cord at the CheckpointAggregator level.
      absl::StatusOr<std::unique_ptr<CheckpointAggregator>> other =
          CheckpointAggregator::Deserialize(&intrinsics_,
                                            std::move(unencrypted_data));
      if (!other.ok()) {
        return ToSessionWriteFinishedResponse(
            absl::Status(other.status().code(),
                         absl::StrCat("Failed to deserialize checkpoint for "
                                      "AGGREGATION_TYPE_MERGE: ",
                                      other.status().message())));
      }
      FCP_RETURN_IF_ERROR(aggregator_->MergeWith(std::move(*other.value())));
      break;
    }
    default:
      return ToSessionWriteFinishedResponse(absl::InvalidArgumentError(
          "AggCoreAggregationType must be specified."));
  }
  return ToSessionWriteFinishedResponse(
      absl::OkStatus(),
      write_request.first_request_metadata().total_size_bytes());
}

// Runs the requested finalization operation and write the uncompressed result
// to the stream. After finalization, the session state is no longer mutable.
//
// Any errors in HandleFinalize kill the stream, since the stream can no longer
// be modified after the Finalize call.
absl::StatusOr<SessionResponse> FedSqlSession::FinalizeSession(
    const FinalizeRequest& request, const BlobMetadata& input_metadata) {
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
  switch (finalize_config.type()) {
    case fcp::confidentialcompute::FINALIZATION_TYPE_REPORT: {
      if (!aggregator_->CanReport()) {
        return absl::FailedPreconditionError(
            "The aggregation can't be completed due to failed preconditions.");
      }
      // Fail if there were no valid inputs, as this likely indicates some issue
      // with configuration of the overall workload.
      FCP_ASSIGN_OR_RETURN(int num_checkpoints_aggregated,
                           aggregator_->GetNumCheckpointsAggregated());
      if (num_checkpoints_aggregated < 1) {
        return absl::InvalidArgumentError(
            "The aggregation can't be successfully completed because no inputs "
            "were aggregated.\n"
            "This may be because inputs were ignored due to an earlier error.");
      }

      // Extract unencrypted checkpoint from the aggregator.
      // Using the scope below ensures that both CheckpointBuilder and Cord
      // are promptly deleted.
      std::string unencrypted_result;
      {
        FederatedComputeCheckpointBuilderFactory builder_factory;
        std::unique_ptr<CheckpointBuilder> checkpoint_builder =
            builder_factory.Create();
        FCP_RETURN_IF_ERROR(aggregator_->Report(*checkpoint_builder));
        aggregator_.reset();
        FCP_ASSIGN_OR_RETURN(absl::Cord checkpoint_cord,
                             checkpoint_builder->Build());
        absl::CopyCordToString(checkpoint_cord, &unencrypted_result);
      }

      if (input_metadata.has_unencrypted() ||
          report_output_access_policy_node_id_ == std::nullopt) {
        result_metadata.set_compression_type(
            BlobMetadata::COMPRESSION_TYPE_NONE);
        result_metadata.set_total_size_bytes(unencrypted_result.size());
        result_metadata.mutable_unencrypted();
        result = std::move(unencrypted_result);
        break;
      }

      FCP_ASSIGN_OR_RETURN(
          std::tie(result_metadata, result),
          EncryptSessionResult(input_metadata, unencrypted_result,
                               *report_output_access_policy_node_id_));
      break;
    }
    case FINALIZATION_TYPE_SERIALIZE: {
      // Serialize the aggregator and bundle it with the private state.
      FCP_ASSIGN_OR_RETURN(std::string serialized_data,
                           std::move(*aggregator_).Serialize());
      aggregator_.reset();
      if (input_metadata.has_unencrypted()) {
        result = std::move(serialized_data);
        result_metadata.set_total_size_bytes(result.size());
        result_metadata.mutable_unencrypted();
        break;
      }
      if (serialize_output_access_policy_node_id_ == std::nullopt) {
        return absl::InvalidArgumentError(
            "No output access policy node ID set for serialized outputs. This "
            "must be set to output serialized state.");
      }
      // Encrypt the bundled blob.
      FCP_ASSIGN_OR_RETURN(
          std::tie(result_metadata, result),
          EncryptSessionResult(input_metadata, serialized_data,
                               *serialize_output_access_policy_node_id_));
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
  return response;
}

absl::Status FedSqlSession::ConfigureSession(
    fcp::confidentialcompute::SessionRequest configure_request) {
  if (!configure_request.configure().has_configuration()) {
    return absl::OkStatus();
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
