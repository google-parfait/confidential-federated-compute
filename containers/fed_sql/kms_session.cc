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

#include <execution>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/fed_sql/sensitive_columns.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/private_state.h"
#include "containers/session.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
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

using ::confidential_federated_compute::sql::SqliteAdapter;
using ::confidential_federated_compute::sql::TensorColumn;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_MERGE;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ColumnConfiguration;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerInitializeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::GemmaInitializeConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::TableSchema;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_INT64;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_STRING;
using ::google::protobuf::Struct;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::DT_DOUBLE;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Intrinsic;
using ::tensorflow_federated::aggregation::kDeltaIndex;
using ::tensorflow_federated::aggregation::kEpsilonIndex;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
}  // namespace

absl::StatusOr<std::unique_ptr<CheckpointParser>>
KmsFedSqlSession::ExecuteClientQuery(const SqlConfiguration& configuration,
                                     CheckpointParser* parser) {
  FCP_ASSIGN_OR_RETURN(
      std::vector<TensorColumn> contents,
      Deserialize(configuration.input_schema, parser,
                  inference_model_->GetInferenceConfiguration()));
  if (inference_model_->HasModel()) {
    FCP_RETURN_IF_ERROR(inference_model_->RunInference(contents));
  }
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<SqliteAdapter> sqlite,
                       SqliteAdapter::Create());
  FCP_RETURN_IF_ERROR(sqlite->DefineTable(configuration.input_schema));
  if (contents.size() > 0) {
    int num_rows = contents.at(0).tensor_.num_elements();
    FCP_RETURN_IF_ERROR(HashSensitiveColumns(contents, sensitive_values_key_));
    FCP_RETURN_IF_ERROR(
        sqlite->AddTableContents(std::move(contents), num_rows));
  }
  FCP_ASSIGN_OR_RETURN(
      std::vector<TensorColumn> result,
      sqlite->EvaluateQuery(configuration.query, configuration.output_columns));
  return std::make_unique<InMemoryCheckpointParser>(std::move(result));
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

  // In case of an error with MergeWith, the session is terminated, since we
  // can't guarantee that the aggregator is in a valid state. If this changes,
  // consider changing this logic to no longer return an error.
  switch (write_config.type()) {
    case AGGREGATION_TYPE_ACCUMULATE: {
      // TODO: Use
      // https://github.com/google-parfait/federated-compute/blob/main/fcp/base/scheduler.h
      // to asynchronously handle deserializing the checkpoint when it is
      // initially written to the session.
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
            ExecuteClientQuery(*sql_configuration_, parser->get());
        if (!sql_result_parser.ok()) {
          return ToSessionWriteFinishedResponse(
              absl::Status(sql_result_parser.status().code(),
                           absl::StrCat("Failed to execute SQL query: ",
                                        sql_result_parser.status().message())));
        }
        parser = std::move(sql_result_parser);
      }

      // Queue the blob so it can be processed on commit.
      uncommitted_inputs_.push_back(
          {.parser = std::move(*parser),
           .metadata = write_request.first_request_metadata()});
      break;
    }
    case AGGREGATION_TYPE_MERGE: {
      //  Merges can be immediately processed.
      FCP_ASSIGN_OR_RETURN(std::unique_ptr<PrivateState> other_private_state,
                           UnbundlePrivateState(unencrypted_data));
      FCP_RETURN_IF_ERROR(private_state_->Merge(*other_private_state));
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

absl::StatusOr<SessionResponse> KmsFedSqlSession::SessionCommit(
    const CommitRequest& commit_request) {
  // In case of an error with Accumulate, the session is terminated, since we
  // can't guarantee that the aggregator is in a valid state. If this changes,
  // consider changing this logic to no longer return an error.
  for (UncommittedInput& uncommitted_input : uncommitted_inputs_) {
    absl::Status accumulate_status =
        aggregator_->Accumulate(*uncommitted_input.parser);
    if (!accumulate_status.ok()) {
      if (absl::IsNotFound(accumulate_status)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Failed to accumulate SQL query results: ",
                         accumulate_status.message()));
      }
      return accumulate_status;
    }
  }
  SessionResponse response = ToSessionCommitResponse(absl::OkStatus());
  uncommitted_inputs_.clear();
  return response;
}

// Runs the requested finalization operation and write the uncompressed result
// to the stream. After finalization, the session state is no longer mutable.
//
// Any errors in HandleFinalize kill the stream, since the stream can no longer
// be modified after the Finalize call.
absl::StatusOr<SessionResponse> KmsFedSqlSession::FinalizeSession(
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

      // Extract unecrypted checkpoint from the aggregator.
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
          Record result_record,
          EncryptSessionResult(input_metadata, unencrypted_result,
                               *report_output_access_policy_node_id_));
      result_metadata = GetBlobMetadataFromRecord(result_record);
      result = std::move(
          *result_record.mutable_hpke_plus_aead_data()->mutable_ciphertext());
      break;
    }
    case FINALIZATION_TYPE_SERIALIZE: {
      // Serialize the aggregator and bundle it with the private state.
      FCP_ASSIGN_OR_RETURN(std::string serialized_data,
                           std::move(*aggregator_).Serialize());
      aggregator_.reset();
      serialized_data = BundlePrivateState(serialized_data, *private_state_);
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
          Record result_record,
          EncryptSessionResult(input_metadata, serialized_data,
                               *serialize_output_access_policy_node_id_));
      result_metadata = GetBlobMetadataFromRecord(result_record);
      result = std::move(
          *result_record.mutable_hpke_plus_aead_data()->mutable_ciphertext());
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
