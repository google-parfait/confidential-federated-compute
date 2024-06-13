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
#include "fcp/protos/confidentialcompute/agg_core_container_config.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/support/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::AggCoreContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::AggCoreContainerWriteConfiguration;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_MERGE;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::ServerContext;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::Configuration;
using ::tensorflow_federated::aggregation::DT_DOUBLE;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Intrinsic;
using ::tensorflow_federated::aggregation::Tensor;

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

SessionResponse ToSessionWriteResponse(absl::Status status,
                                       long available_memory) {
  grpc::Status grpc_status = ToGrpcStatus(std::move(status));
  SessionResponse session_response;
  WriteFinishedResponse* response = session_response.mutable_write();
  response->mutable_status()->set_code(grpc_status.error_code());
  response->mutable_status()->set_message(grpc_status.error_message());
  response->set_write_capacity_bytes(available_memory);
  return session_response;
}

// Decrypts and parses a record and accumulates it into the state of the
// CheckpointAggregator `aggregator`.
//
// Returns an error if the aggcore state may be invalid and the session needs to
// be shut down. Otherwise, reports status to the client in
// WriteFinishedResponse
//
// TODO: handle blobs that span multiple WriteRequests.
// TODO: add tracking for available memory.
absl::Status HandleWrite(
    const WriteRequest& request, CheckpointAggregator& aggregator,
    BlobDecryptor* blob_decryptor, SessionNonceTracker& nonce_tracker,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream,
    long available_memory, const std::vector<Intrinsic>* intrinsics) {
  if (absl::Status nonce_status =
          nonce_tracker.CheckBlobNonce(request.first_request_metadata());
      !nonce_status.ok()) {
    stream->Write(ToSessionWriteResponse(nonce_status, available_memory));
    return absl::OkStatus();
  }

  AggCoreContainerWriteConfiguration write_config;
  if (!request.first_request_configuration().UnpackTo(&write_config)) {
    stream->Write(ToSessionWriteResponse(
        absl::InvalidArgumentError(
            "Failed to parse AggCoreContainerWriteConfiguration."),
        available_memory));
    return absl::OkStatus();
  }
  absl::StatusOr<std::string> unencrypted_data = blob_decryptor->DecryptBlob(
      request.first_request_metadata(), request.data());
  if (!unencrypted_data.ok()) {
    stream->Write(
        ToSessionWriteResponse(unencrypted_data.status(), available_memory));
    return absl::OkStatus();
  }

  // In case of an error with Accumulate or MergeWith, the session is
  // terminated, since we can't guarantee that the aggregator is in a valid
  // state. If this changes, consider changing this logic to no longer return an
  // error.
  switch (write_config.type()) {
    case AGGREGATION_TYPE_ACCUMULATE: {
      FederatedComputeCheckpointParserFactory parser_factory;
      absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
          parser_factory.Create(absl::Cord(std::move(*unencrypted_data)));
      if (!parser.ok()) {
        stream->Write(ToSessionWriteResponse(
            absl::Status(parser.status().code(),
                         absl::StrCat("Failed to deserialize checkpoint for "
                                      "AGGREGATION_TYPE_ACCUMULATE: ",
                                      parser.status().message())),
            available_memory));
        return absl::OkStatus();
      }
      FCP_RETURN_IF_ERROR(aggregator.Accumulate(*parser.value()));
      break;
    }
    case AGGREGATION_TYPE_MERGE: {
      absl::StatusOr<std::unique_ptr<CheckpointAggregator>> other =
          CheckpointAggregator::Deserialize(intrinsics, *unencrypted_data);
      if (!other.ok()) {
        stream->Write(ToSessionWriteResponse(
            absl::Status(other.status().code(),
                         absl::StrCat("Failed to deserialize checkpoint for "
                                      "AGGREGATION_TYPE_MERGE: ",
                                      other.status().message())),
            available_memory));
        return absl::OkStatus();
      }
      FCP_RETURN_IF_ERROR(aggregator.MergeWith(std::move(*other.value())));
      break;
    }
    default:
      stream->Write(ToSessionWriteResponse(
          absl::InvalidArgumentError(
              "AggCoreAggregationType must be specified."),
          available_memory));
      return absl::OkStatus();
  }

  SessionResponse session_response;
  WriteFinishedResponse* response = session_response.mutable_write();
  response->set_committed_size_bytes(
      request.first_request_metadata().total_size_bytes());
  response->mutable_status()->set_code(grpc::OK);
  response->set_write_capacity_bytes(available_memory);

  stream->Write(session_response);
  return absl::OkStatus();
}

// Runs the requested finalization operation and write the uncompressed result
// to the stream. After finalization, the session state is no longer mutable.
//
// Any errors in HandleFinalize kill the stream, since the stream can no longer
// be modified after the Finalize call.
absl::Status HandleFinalize(
    const FinalizeRequest& request,
    std::unique_ptr<CheckpointAggregator> aggregator,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream,
    const BlobMetadata& input_metadata) {
  AggCoreContainerFinalizeConfiguration finalize_config;
  if (!request.configuration().UnpackTo(&finalize_config)) {
    return absl::InvalidArgumentError(
        "Failed to parse AggCoreContainerFinalizeConfiguration.");
  }
  std::string result;
  BlobMetadata result_metadata;
  switch (finalize_config.type()) {
    case fcp::confidentialcompute::FINALIZATION_TYPE_REPORT: {
      if (!aggregator->CanReport()) {
        return absl::FailedPreconditionError(
            "The aggregation can't be completed due to failed preconditions.");
      }

      FederatedComputeCheckpointBuilderFactory builder_factory;
      std::unique_ptr<CheckpointBuilder> checkpoint_builder =
          builder_factory.Create();
      FCP_RETURN_IF_ERROR(aggregator->Report(*checkpoint_builder));
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
                           std::move(*aggregator).Serialize());
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
  stream->Write(response);
  return absl::OkStatus();
}

}  // namespace

absl::Status FedSqlConfidentialTransform::FedSqlInitialize(
    const fcp::confidentialcompute::InitializeRequest* request,
    fcp::confidentialcompute::InitializeResponse* response) {
  // TODO: Switch to using a FedSql-specific wrapper message that contains a
  // aggcore Configuration.
  Configuration config;
  if (!request->configuration().UnpackTo(&config)) {
    return absl::InvalidArgumentError("Configuration cannot be unpacked.");
  }
  FCP_RETURN_IF_ERROR(CheckpointAggregator::ValidateConfig(config));
  const BlobDecryptor* blob_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ != std::nullopt) {
      return absl::FailedPreconditionError(
          "Initialize can only be called once.");
    }

    FCP_ASSIGN_OR_RETURN(
        std::vector<Intrinsic> intrinsics,
        tensorflow_federated::aggregation::ParseFromConfig(config));
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

    intrinsics_.emplace(std::move(intrinsics));
    blob_decryptor_.emplace(crypto_stub_, config_properties);

    // Since blob_decryptor_ is set once in Initialize and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    blob_decryptor = &*blob_decryptor_;
  }

  FCP_ASSIGN_OR_RETURN(*response->mutable_public_key(),
                       blob_decryptor->GetPublicKey());
  return absl::OkStatus();
}

absl::Status FedSqlConfidentialTransform::FedSqlSession(
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream,
    long available_memory) {
  BlobDecryptor* blob_decryptor;
  std::unique_ptr<CheckpointAggregator> aggregator;
  const std::vector<Intrinsic>* intrinsics;
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ == std::nullopt || blob_decryptor_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "Initialize must be called before Session.");
    }

    // Since blob_decryptor_ is set once in Initialize and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that values have been set for the std::optional wrappers.
    blob_decryptor = &*blob_decryptor_;
    intrinsics = &*intrinsics_;
  }

  FCP_ASSIGN_OR_RETURN(aggregator, CheckpointAggregator::Create(intrinsics));
  SessionRequest configure_request;
  bool success = stream->Read(&configure_request);
  if (!success) {
    return absl::AbortedError("Session failed to read client message.");
  }

  if (!configure_request.has_configure()) {
    return absl::FailedPreconditionError(
        "Session must be configured with a ConfigureRequest before any other "
        "requests.");
  }
  SessionResponse configure_response;
  SessionNonceTracker nonce_tracker;
  *configure_response.mutable_configure()->mutable_nonce() =
      nonce_tracker.GetSessionNonce();
  configure_response.mutable_configure()->set_write_capacity_bytes(
      available_memory);
  stream->Write(configure_response);

  // Initialze result_blob_metadata with unencrypted metadata since
  // EarliestExpirationTimeMetadata expects inputs to have either unencrypted or
  // hpke_plus_aead_data.
  BlobMetadata result_blob_metadata;
  result_blob_metadata.mutable_unencrypted();
  SessionRequest session_request;
  while (stream->Read(&session_request)) {
    switch (session_request.kind_case()) {
      case SessionRequest::kWrite: {
        const WriteRequest& write_request = session_request.write();
        // If any of the input blobs are encrypted, then encrypt the result of
        // FINALIZATION_TYPE_SERIALIZE. Use the metadata with the earliest
        // expiration timestamp.
        absl::StatusOr<BlobMetadata> earliest_expiration_metadata =
            EarliestExpirationTimeMetadata(
                result_blob_metadata, write_request.first_request_metadata());
        if (!earliest_expiration_metadata.ok()) {
          stream->Write(ToSessionWriteResponse(
              absl::Status(
                  earliest_expiration_metadata.status().code(),
                  absl::StrCat(
                      "Failed to extract expiration timestamp from "
                      "BlobMetadata with encryption: ",
                      earliest_expiration_metadata.status().message())),
              available_memory));
          break;
        }
        result_blob_metadata = *earliest_expiration_metadata;
        // TODO: spin up a thread to incorporate each blob.
        FCP_RETURN_IF_ERROR(HandleWrite(write_request, *aggregator,
                                        blob_decryptor, nonce_tracker, stream,
                                        available_memory, intrinsics));
        break;
      }
      case SessionRequest::kFinalize:
        return HandleFinalize(session_request.finalize(), std::move(aggregator),
                              stream, result_blob_metadata);
      case SessionRequest::kConfigure:
      default:
        return absl::FailedPreconditionError(absl::StrCat(
            "Session expected a write request but received request of type: ",
            session_request.kind_case()));
    }
  }

  return absl::AbortedError(
      "Session failed to read client write or finalize message.");
}

grpc::Status FedSqlConfidentialTransform::Initialize(
    ServerContext* context, const InitializeRequest* request,
    InitializeResponse* response) {
  return ToGrpcStatus(FedSqlInitialize(request, response));
}

grpc::Status FedSqlConfidentialTransform::Session(
    ServerContext* context,
    grpc::ServerReaderWriter<SessionResponse, SessionRequest>* stream) {
  long available_memory = session_tracker_.AddSession();
  if (available_memory > 0) {
    grpc::Status status = ToGrpcStatus(FedSqlSession(stream, available_memory));
    absl::Status remove_session = session_tracker_.RemoveSession();
    if (!remove_session.ok()) {
      return ToGrpcStatus(remove_session);
    }
    return status;
  }
  return ToGrpcStatus(absl::UnavailableError("No session memory available."));
}

}  // namespace confidential_federated_compute::fed_sql
