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
#include "containers/fed_sql/confidential_transform_server.h"

#include <execution>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/fed_sql/kms_session.h"
#include "containers/fed_sql/sensitive_columns.h"
#include "containers/fed_sql/session_utils.h"
#include "containers/private_state.h"
#include "containers/session.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/support/status.h"
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
using ::fcp::confidentialcompute::FedSqlContainerConfigConstraints;
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

absl::Status ValidateFedSqlOuterDpParameters(const Intrinsic& intrinsic) {
  if (intrinsic.parameters.size() < 2) {
    return absl::InvalidArgumentError(
        "Outer DP IntrinsicConfig must have at least two "
        "parameters.");
  }
  if (intrinsic.parameters.at(kEpsilonIndex).dtype() != DT_DOUBLE ||
      intrinsic.parameters.at(kDeltaIndex).dtype() != DT_DOUBLE) {
    return absl::InvalidArgumentError(
        "Epsilon and delta parameters for outer DP IntrinsicConfig must both "
        "have type DT_DOUBLE.");
  }
  if (intrinsic.parameters.at(kEpsilonIndex).num_elements() != 1 ||
      intrinsic.parameters.at(kDeltaIndex).num_elements() != 1) {
    return absl::InvalidArgumentError(
        "Epsilon and delta parameters for outer DP IntrinsicConfig must each "
        "have exactly one value.");
  }
  return absl::OkStatus();
}

absl::Status ValidateTopLevelIntrinsics(
    const std::vector<Intrinsic>& intrinsics) {
  if (intrinsics.size() != 1) {
    return absl::InvalidArgumentError(
        "Configuration must have exactly one IntrinsicConfig.");
  }
  return absl::OkStatus();
}

std::string CreateTempFilePath(std::string directory,
                               std::string basename_prefix,
                               uint32_t basename_id) {
  // Created temp file will be in <directory>/<basename_prefix>_<basename_id>.
  return absl::StrCat(directory, "/", basename_prefix, "_",
                      std::to_string(basename_id));
}

absl::Status AppendBytesToTempFile(std::string& file_path,
                                   std::ios_base::openmode mode,
                                   const char* data,
                                   std::streamsize data_size) {
  // Write or append binary content to file depending on mode.
  std::ofstream temp_file(file_path, mode);
  if (!temp_file.is_open()) {
    return absl::DataLossError(
        absl::StrCat("Failed to open temp file for writing: ", file_path));
  }
  temp_file.write(data, data_size);
  temp_file.close();
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<CheckpointParser>>
FedSqlSession::ExecuteClientQuery(const SqlConfiguration& configuration,
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
            ExecuteClientQuery(*sql_configuration_, parser->get());
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
      FCP_ASSIGN_OR_RETURN(std::unique_ptr<PrivateState> other_private_state,
                           UnbundlePrivateState(unencrypted_data));
      FCP_RETURN_IF_ERROR(private_state_->Merge(*other_private_state));
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

absl::Status FedSqlConfidentialTransform::SetAndValidateIntrinsics(
    const FedSqlContainerInitializeConfiguration& config) {
  absl::MutexLock l(&mutex_);
  if (intrinsics_ != std::nullopt) {
    return absl::FailedPreconditionError(
        "SetIntrinsics can only be called once.");
  }
  FCP_RETURN_IF_ERROR(
      CheckpointAggregator::ValidateConfig(config.agg_configuration()));

  FCP_ASSIGN_OR_RETURN(std::vector<Intrinsic> intrinsics,
                       tensorflow_federated::aggregation::ParseFromConfig(
                           config.agg_configuration()));
  FCP_RETURN_IF_ERROR(ValidateTopLevelIntrinsics(intrinsics));
  const Intrinsic& fedsql_intrinsic = intrinsics.at(0);
  if (fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPGroupByUri ||
      fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPTensorAggregatorBundleUri) {
    FCP_RETURN_IF_ERROR(ValidateFedSqlOuterDpParameters(fedsql_intrinsic));
  }

  intrinsics_.emplace(std::move(intrinsics));
  return absl::OkStatus();
}

absl::StatusOr<Struct> FedSqlConfidentialTransform::GetConfigConstraints(
    const FedSqlContainerInitializeConfiguration& config) {
  const std::vector<Intrinsic>* intrinsics;
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "Intrinsics have not been initialized.");
    }
    intrinsics = &*intrinsics_;
  }

  const Intrinsic& fedsql_intrinsic = intrinsics->at(0);
  Struct config_properties;
  (*config_properties.mutable_fields())["intrinsic_uri"].set_string_value(
      fedsql_intrinsic.uri);
  if (fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPGroupByUri ||
      fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPTensorAggregatorBundleUri) {
    double epsilon =
        fedsql_intrinsic.parameters.at(kEpsilonIndex).CastToScalar<double>();
    double delta =
        fedsql_intrinsic.parameters.at(kDeltaIndex).CastToScalar<double>();
    (*config_properties.mutable_fields())["epsilon"].set_number_value(epsilon);
    (*config_properties.mutable_fields())["delta"].set_number_value(delta);
  }
  if (config.serialize_output_access_policy_node_id() > 0) {
    (*config_properties.mutable_fields())["serialize_dest"].set_number_value(
        config.serialize_output_access_policy_node_id());
    serialize_output_access_policy_node_id_.emplace(
        config.serialize_output_access_policy_node_id());
  }
  if (config.report_output_access_policy_node_id() > 0) {
    (*config_properties.mutable_fields())["report_dest"].set_number_value(
        config.report_output_access_policy_node_id());
    report_output_access_policy_node_id_.emplace(
        config.report_output_access_policy_node_id());
  }
  return config_properties;
}

absl::Status FedSqlConfidentialTransform::ValidateConfigConstraints(
    const FedSqlContainerConfigConstraints& config_constraints) {
  const std::vector<Intrinsic>* intrinsics;
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "Intrinsics have not been initialized.");
    }
    intrinsics = &*intrinsics_;
  }

  const Intrinsic& fedsql_intrinsic = intrinsics->at(0);
  if (fedsql_intrinsic.uri != config_constraints.intrinsic_uri()) {
    return absl::FailedPreconditionError(
        "Invalid intrinsic URI for DP configuration.");
  }

  if (fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPGroupByUri ||
      fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPTensorAggregatorBundleUri) {
    double epsilon =
        fedsql_intrinsic.parameters.at(kEpsilonIndex).CastToScalar<double>();
    if (config_constraints.epsilon() != epsilon) {
      return absl::FailedPreconditionError(
          "Epsilon value does not match the expected value.");
    }

    double delta =
        fedsql_intrinsic.parameters.at(kDeltaIndex).CastToScalar<double>();
    if (config_constraints.delta() != delta) {
      return absl::FailedPreconditionError(
          "Delta value does not match the expected value.");
    }
  }
  return absl::OkStatus();
}

absl::Status FedSqlConfidentialTransform::InitializeInferenceModel(
    const InferenceInitializeConfiguration& inference_init_config) {
  // Check that all data blobs passed in through WriteConfigurationRequest
  // are committed.
  for (const auto& [config_id, config_metadata] : write_configuration_map_) {
    if (!config_metadata.commit) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Data blob with configuration_id ", config_id, " is not committed."));
    }
  }

  if (inference_init_config.model_init_config_case() ==
      InferenceInitializeConfiguration::MODEL_INIT_CONFIG_NOT_SET) {
    return absl::FailedPreconditionError(
        "When FedSqlContainerInitializeConfiguration.inference_init_config is "
        "set, InferenceInitializeConfiguration.model_init_config must be set.");
  }
  if (inference_init_config.inference_config().model_config_case() ==
      fcp::confidentialcompute::InferenceConfiguration::MODEL_CONFIG_NOT_SET) {
    return absl::FailedPreconditionError(
        "When FedSqlContainerInitializeConfiguration.inference_init_config is "
        "set, InferenceConfiguration.model_config must be set.");
  }
  for (const fcp::confidentialcompute::InferenceTask& inference_task :
       inference_init_config.inference_config().inference_task()) {
    if (inference_task.inference_logic_case() ==
        fcp::confidentialcompute::InferenceTask::INFERENCE_LOGIC_NOT_SET) {
      return absl::FailedPreconditionError(
          "When FedSqlContainerInitializeConfiguration.inference_init_config "
          "is set, InferenceConfiguration.inference_task.inference_logic must "
          "be set for all inference tasks.");
    }
  }

  // Populate inference_configuration_.initialize_configuration.
  inference_configuration_.emplace();
  inference_configuration_->initialize_configuration = inference_init_config;

  switch (inference_configuration_->initialize_configuration
              .model_init_config_case()) {
    case InferenceInitializeConfiguration::kGemmaInitConfig: {
      const GemmaInitializeConfiguration& gemma_init_config =
          inference_configuration_->initialize_configuration
              .gemma_init_config();
      if (write_configuration_map_.find(
              gemma_init_config.tokenizer_configuration_id()) ==
          write_configuration_map_.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected Gemma tokenizer configuration id ",
                         gemma_init_config.tokenizer_configuration_id(),
                         " is missing in WriteConfigurationRequest."));
      }
      if (write_configuration_map_.find(
              gemma_init_config.model_weight_configuration_id()) ==
          write_configuration_map_.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected Gemma model weight configuration id ",
                         gemma_init_config.model_weight_configuration_id(),
                         " is missing in WriteConfigurationRequest."));
      }
      // Populate inference_configuration_.gemma_configuration.
      SessionGemmaConfiguration gemma_config;
      gemma_config.tokenizer_path =
          write_configuration_map_[gemma_init_config
                                       .tokenizer_configuration_id()]
              .file_path;
      gemma_config.model_weight_path =
          write_configuration_map_[gemma_init_config
                                       .model_weight_configuration_id()]
              .file_path;
      inference_configuration_->gemma_configuration = std::move(gemma_config);
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported model_init_config_case: ",
                       inference_configuration_->initialize_configuration
                           .model_init_config_case()));
  }

  return inference_model_->BuildModel(*inference_configuration_);
}

absl::Status FedSqlConfidentialTransform::StreamInitializeTransformWithKms(
    const ::google::protobuf::Any& configuration,
    const ::google::protobuf::Any& config_constraints,
    std::vector<std::string> reencryption_keys) {
  FedSqlContainerInitializeConfiguration fed_sql_config;
  if (!configuration.UnpackTo(&fed_sql_config)) {
    return absl::InvalidArgumentError(
        "FedSqlContainerInitializeConfiguration cannot be unpacked.");
  }
  FedSqlContainerConfigConstraints fed_sql_config_constraints;
  if (!config_constraints.UnpackTo(&fed_sql_config_constraints)) {
    return absl::InvalidArgumentError(
        "FedSqlContainerConfigConstraints cannot be unpacked.");
  }
  FCP_RETURN_IF_ERROR(SetAndValidateIntrinsics(fed_sql_config));
  FCP_RETURN_IF_ERROR(ValidateConfigConstraints(fed_sql_config_constraints));
  if (fed_sql_config.has_inference_init_config()) {
    FCP_RETURN_IF_ERROR(
        InitializeInferenceModel(fed_sql_config.inference_init_config()));
  }
  reencryption_keys_ = std::move(reencryption_keys);
  return absl::OkStatus();
}

absl::StatusOr<Struct> FedSqlConfidentialTransform::StreamInitializeTransform(
    const InitializeRequest* request) {
  FedSqlContainerInitializeConfiguration config;
  if (!request->configuration().UnpackTo(&config)) {
    return absl::InvalidArgumentError(
        "FedSqlContainerInitializeConfiguration cannot be unpacked.");
  }
  FCP_RETURN_IF_ERROR(SetAndValidateIntrinsics(config));
  FCP_ASSIGN_OR_RETURN(Struct config_properties, GetConfigConstraints(config));
  if (config.has_inference_init_config()) {
    FCP_RETURN_IF_ERROR(
        InitializeInferenceModel(config.inference_init_config()));
  }
  return config_properties;
}

absl::Status FedSqlConfidentialTransform::ReadWriteConfigurationRequest(
    const fcp::confidentialcompute::WriteConfigurationRequest&
        write_configuration) {
  std::ios_base::openmode file_open_mode;
  // First request metadata is set for the first WriteConfigurationRequest of a
  // new data blob.
  if (write_configuration.has_first_request_metadata()) {
    // Create a new file.
    file_open_mode = std::ios::binary;
    current_configuration_id_ =
        write_configuration.first_request_metadata().configuration_id();
    if (write_configuration_map_.find(current_configuration_id_) !=
        write_configuration_map_.end()) {
      return absl::InvalidArgumentError(
          "Duplicated configuration_id found in WriteConfigurationRequest.");
    }
    // Create a new temp files. Temp files are saved as
    // /tmp/write_configuration_1, /tmp/write_configuration_2, etc. Use
    // `write_configuration_map_.size() + 1` to distinguish different temp file
    // names.
    std::string temp_file_path = CreateTempFilePath(
        "/tmp", "write_configuration", write_configuration_map_.size() + 1);

    LOG(INFO) << "Start writing bytes for configuration_id: "
              << current_configuration_id_ << " to " << temp_file_path;

    write_configuration_map_[current_configuration_id_] =
        WriteConfigurationMetadata{
            .file_path = std::move(temp_file_path),
            .total_size_bytes = static_cast<uint64_t>(
                write_configuration.first_request_metadata()
                    .total_size_bytes()),
            .commit = write_configuration.commit()};
  } else {
    // If the current write_configuration is not the first
    // WriteConfigurationRequest of a data blob, append to existing file.
    file_open_mode = std::ios::binary | std::ios::app;
  }

  auto& [current_file_path, current_total_size_bytes, commit] =
      write_configuration_map_[current_configuration_id_];
  FCP_RETURN_IF_ERROR(AppendBytesToTempFile(current_file_path, file_open_mode,
                                            write_configuration.data().data(),
                                            write_configuration.data().size()));
  // Update the commit status of the data blob in write_configuration_map_.
  commit = write_configuration.commit();

  // When it's the last WriteConfigurationRequest of a blob, check the size of
  // the file matches the expectation.
  if (commit) {
    if (std::filesystem::file_size(current_file_path) !=
        current_total_size_bytes) {
      return absl::InvalidArgumentError(
          absl::StrCat("The total size of the data blob does not match "
                       "expected size. Expecting ",
                       current_total_size_bytes, ", got ",
                       std::filesystem::file_size(current_file_path)));
    }
    LOG(INFO) << "Successfully wrote all " << current_total_size_bytes
              << " bytes to " << current_file_path;
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
FedSqlConfidentialTransform::CreateSession() {
  std::unique_ptr<CheckpointAggregator> aggregator;
  const std::vector<Intrinsic>* intrinsics;
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "Initialize must be called before Session.");
    }

    // Since intrinsics_ is set once in Initialize and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that values have been set for the std::optional wrappers.
    intrinsics = &*intrinsics_;
  }

  FCP_ASSIGN_OR_RETURN(aggregator, CheckpointAggregator::Create(intrinsics));
  if (KmsEnabled()) {
    return std::make_unique<KmsFedSqlSession>(
        std::move(aggregator), *intrinsics, inference_model_,
        serialize_output_access_policy_node_id_,
        report_output_access_policy_node_id_, sensitive_values_key_);
  } else {
    return std::make_unique<FedSqlSession>(
        std::move(aggregator), *intrinsics, inference_model_,
        serialize_output_access_policy_node_id_,
        report_output_access_policy_node_id_, sensitive_values_key_);
  }
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
