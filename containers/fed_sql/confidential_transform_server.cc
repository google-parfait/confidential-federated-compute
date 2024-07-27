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
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/support/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_MERGE;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerInitializeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::SessionResponse;
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

constexpr char kFedSqlDpGroupByUri[] = "fedsql_dp_group_by";

absl::Status ValidateFedSqlDpGroupByParameters(const Intrinsic& intrinsic) {
  if (intrinsic.parameters.size() < 2) {
    return absl::InvalidArgumentError(
        "`fedsql_dp_group_by` IntrinsicConfig must have at least two "
        "parameters.");
  }
  if (intrinsic.parameters.at(0).dtype() != DT_DOUBLE ||
      intrinsic.parameters.at(1).dtype() != DT_DOUBLE) {
    return absl::InvalidArgumentError(
        "`fedsql_dp_group_by` parameters must both have type DT_DOUBLE.");
  }
  if (intrinsic.parameters.at(0).num_elements() != 1 ||
      intrinsic.parameters.at(1).num_elements() != 1) {
    return absl::InvalidArgumentError(
        "`fedsql_dp_group_by` parameters must each have exactly one value.");
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

}  // namespace

absl::StatusOr<SessionResponse> FedSqlSession::SessionWrite(
    const WriteRequest& write_request, std::string unencrypted_data) {
  FedSqlContainerWriteConfiguration write_config;
  if (!write_request.first_request_configuration().UnpackTo(&write_config)) {
    return ToSessionWriteFinishedResponse(absl::InvalidArgumentError(
        "Failed to parse FedSqlContainerWriteConfiguration."));
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
      FCP_RETURN_IF_ERROR(aggregator_->Accumulate(*parser.value()));
      break;
    }
    case AGGREGATION_TYPE_MERGE: {
      absl::StatusOr<std::unique_ptr<CheckpointAggregator>> other =
          CheckpointAggregator::Deserialize(&intrinsics_, unencrypted_data);
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

      FederatedComputeCheckpointBuilderFactory builder_factory;
      std::unique_ptr<CheckpointBuilder> checkpoint_builder =
          builder_factory.Create();
      FCP_RETURN_IF_ERROR(aggregator_->Report(*checkpoint_builder));
      FCP_ASSIGN_OR_RETURN(absl::Cord checkpoint_cord,
                           checkpoint_builder->Build());
      absl::CopyCordToString(checkpoint_cord, &result);
      result_metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
      result_metadata.set_total_size_bytes(result.size());
      result_metadata.mutable_unencrypted();
      break;
    }
    case FINALIZATION_TYPE_SERIALIZE: {
      FCP_ASSIGN_OR_RETURN(std::string serialized_aggregator,
                           std::move(*aggregator_).Serialize());
      if (input_metadata.has_unencrypted()) {
        result = std::move(serialized_aggregator);
        result_metadata.set_total_size_bytes(result.size());
        result_metadata.mutable_unencrypted();
        break;
      }
      RecordEncryptor encryptor;
      BlobHeader previous_header;
      if (!previous_header.ParseFromString(input_metadata.hpke_plus_aead_data()
                                               .ciphertext_associated_data())) {
        return absl::InvalidArgumentError(
            "Failed to parse the BlobHeader when trying to encrypt outputs.");
      }
      FCP_ASSIGN_OR_RETURN(Record result_record,
                           encryptor.EncryptRecord(
                               serialized_aggregator,
                               input_metadata.hpke_plus_aead_data()
                                   .rewrapped_symmetric_key_associated_data()
                                   .reencryption_public_key(),
                               previous_header.access_policy_sha256(),
                               finalize_config.output_access_policy_node_id()));
      result_metadata = GetBlobMetadataFromRecord(result_record);
      result = result_record.hpke_plus_aead_data().ciphertext();
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Finalize configuration must specify the finalization type.");
  }

  SessionResponse response;
  ReadResponse* read_response = response.mutable_read();
  read_response->set_finish_read(true);
  *(read_response->mutable_data()) = result;
  *(read_response->mutable_first_response_metadata()) = result_metadata;
  return response;
}

absl::StatusOr<google::protobuf::Struct>
FedSqlConfidentialTransform::InitializeTransform(
    const fcp::confidentialcompute::InitializeRequest* request) {
  FedSqlContainerInitializeConfiguration config;
  if (!request->configuration().UnpackTo(&config)) {
    return absl::InvalidArgumentError(
        "FedSqlContainerInitializeConfiguration cannot be unpacked.");
  }
  FCP_RETURN_IF_ERROR(
      CheckpointAggregator::ValidateConfig(config.agg_configuration()));
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ != std::nullopt) {
      return absl::FailedPreconditionError(
          "Initialize can only be called once.");
    }

    FCP_ASSIGN_OR_RETURN(std::vector<Intrinsic> intrinsics,
                         tensorflow_federated::aggregation::ParseFromConfig(
                             config.agg_configuration()));
    FCP_RETURN_IF_ERROR(ValidateTopLevelIntrinsics(intrinsics));
    google::protobuf::Struct config_properties;
    (*config_properties.mutable_fields())["intrinsic_uri"].set_string_value(
        intrinsics.at(0).uri);
    if (intrinsics.at(0).uri == kFedSqlDpGroupByUri) {
      const Intrinsic& fedsql_dp_intrinsic = intrinsics.at(0);
      FCP_RETURN_IF_ERROR(
          ValidateFedSqlDpGroupByParameters(fedsql_dp_intrinsic));
      double epsilon =
          fedsql_dp_intrinsic.parameters.at(0).CastToScalar<double>();
      double delta =
          fedsql_dp_intrinsic.parameters.at(1).CastToScalar<double>();
      (*config_properties.mutable_fields())["epsilon"].set_number_value(
          epsilon);
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

    intrinsics_.emplace(std::move(intrinsics));
    return config_properties;
  }
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
  return std::make_unique<FedSqlSession>(
      FedSqlSession(std::move(aggregator), *intrinsics));
}
}  // namespace confidential_federated_compute::fed_sql
