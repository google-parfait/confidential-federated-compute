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
#include "containers/tff_server/confidential_transform_server.h"

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
#include "fcp/confidentialcompute/tff_execution_helper.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/tff_config.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/support/status.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/converters.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

namespace confidential_federated_compute::tff_server {

using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::TffSessionConfig;
using ::fcp::confidentialcompute::TffSessionWriteConfig;
using ::fcp::confidentialcompute::WriteRequest;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;

absl::Status TffSession::ConfigureSession(
    fcp::confidentialcompute::SessionRequest configure_request) {
  if (child_executor_ != nullptr) {
    return absl::FailedPreconditionError("Session already configured.");
  }

  TffSessionConfig session_config;
  if (!configure_request.has_configure() ||
      !configure_request.configure().configuration().UnpackTo(
          &session_config)) {
    return absl::InvalidArgumentError("TffSessionConfig invalid.");
  }

  auto leaf_executor_fn = []() {
    return tensorflow_federated::CreateReferenceResolvingExecutor(
        tensorflow_federated::CreateTensorFlowExecutor());
  };
  tensorflow_federated::CardinalityMap cardinality_map;
  cardinality_map[tensorflow_federated::kClientsUri] =
      session_config.num_clients();
  FCP_ASSIGN_OR_RETURN(
      auto federating_executor,
      tensorflow_federated::CreateFederatingExecutor(
          /*server_child=*/leaf_executor_fn(),
          /*client_child=*/leaf_executor_fn(), cardinality_map));
  child_executor_ = tensorflow_federated::CreateReferenceResolvingExecutor(
      federating_executor);
  function_ = std::move(session_config.function());
  if (session_config.has_initial_arg()) {
    argument_ = std::move(session_config.initial_arg());
  }
  output_access_policy_node_id_ = session_config.output_access_policy_node_id();

  return absl::OkStatus();
}

absl::StatusOr<SessionResponse> TffSession::ParseData(
    const std::string& uri, std::string unencrypted_data,
    int64_t total_size_bytes) {
  tensorflow_federated::v0::Value value;
  if (!value.ParseFromString(unencrypted_data)) {
    return ToSessionWriteFinishedResponse(absl::InvalidArgumentError(
        "Failed to deserialize the data to a TFF Value."));
  }
  auto [it, inserted] = data_by_uri_.insert({uri, std::move(value)});
  if (!inserted) {
    return ToSessionWriteFinishedResponse(absl::FailedPreconditionError(
        "Data corresponding to URI already written to session."));
  }
  return ToSessionWriteFinishedResponse(absl::OkStatus(), total_size_bytes);
}

absl::StatusOr<SessionResponse> TffSession::ParseClientData(
    const std::string& uri, std::string unencrypted_data,
    int64_t total_size_bytes) {
  FederatedComputeCheckpointParserFactory parser_factory;
  absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
      parser_factory.Create(absl::Cord(std::move(unencrypted_data)));
  if (!parser.ok()) {
    return ToSessionWriteFinishedResponse(absl::Status(
        parser.status().code(),
        absl::StrCat("Failed to deserialize the federated compute checkpoint. ",
                     parser.status().message())));
  }
  auto [it, inserted] =
      client_checkpoint_parser_by_uri_.insert({uri, std::move(parser.value())});
  if (!inserted) {
    return ToSessionWriteFinishedResponse(absl::FailedPreconditionError(
        "Data corresponding to URI already written to session."));
  }
  return ToSessionWriteFinishedResponse(absl::OkStatus(), total_size_bytes);
}

absl::StatusOr<SessionResponse> TffSession::SessionWrite(
    const WriteRequest& write_request, std::string unencrypted_data) {
  if (child_executor_ == nullptr) {
    return absl::FailedPreconditionError(
        "Session must be configured before data can be written.");
  }

  TffSessionWriteConfig write_config;
  if (!write_request.has_first_request_configuration() ||
      !write_request.first_request_configuration().UnpackTo(&write_config)) {
    return ToSessionWriteFinishedResponse(
        absl::InvalidArgumentError("Failed to parse TffSessionWriteConfig."));
  }

  if (write_config.client_upload()) {
    return ParseClientData(
        write_config.uri(), std::move(unencrypted_data),
        write_request.first_request_metadata().total_size_bytes());
  }

  return ParseData(write_config.uri(), std::move(unencrypted_data),
                   write_request.first_request_metadata().total_size_bytes());
}

absl::StatusOr<tensorflow_federated::v0::Value> TffSession::FetchData(
    const std::string& uri) {
  auto data = data_by_uri_.find(uri);
  if (data == data_by_uri_.end()) {
    return absl::InvalidArgumentError(
        "Data in argument was not provided to the transform.");
  }
  return data->second;
}

absl::StatusOr<tensorflow_federated::v0::Value> TffSession::FetchClientData(
    const std::string& uri, const std::string& key) {
  auto parser = client_checkpoint_parser_by_uri_.find(uri);
  if (parser == client_checkpoint_parser_by_uri_.end()) {
    return absl::InvalidArgumentError(
        "Data in argument was not provided to the transform.");
  }
  // Note that each key can only be accessed a single time from the parser. So,
  // this relies on the fact that a given uri, key pair will only appear once in
  // the input argument.
  absl::StatusOr<tensorflow_federated::aggregation::Tensor> agg_tensor =
      parser->second->GetTensor(key);
  if (!agg_tensor.ok()) {
    return absl::Status(
        agg_tensor.status().code(),
        absl::StrCat("Invalid tensor name. ", agg_tensor.status().message()));
  }
  absl::StatusOr<tensorflow::Tensor> tensor =
      tensorflow_federated::aggregation::tensorflow::ToTfTensor(
          std::move(*agg_tensor));
  if (!tensor.ok()) {
    return absl::Status(
        tensor.status().code(),
        absl::StrCat("Invalid tensor data. ", tensor.status().message()));
  }
  tensorflow_federated::v0::Value value;
  FCP_RETURN_IF_ERROR(
      tensorflow_federated::SerializeTensorValue(std::move(*tensor), &value));

  return value;
}

absl::StatusOr<SessionResponse> TffSession::FinalizeSession(
    const FinalizeRequest& request, const BlobMetadata& input_metadata) {
  if (child_executor_ == nullptr) {
    return absl::FailedPreconditionError(
        "Session must be configured before it can be finalized.");
  }

  FCP_ASSIGN_OR_RETURN(tensorflow_federated::OwnedValueId fn_handle,
                       child_executor_->CreateValue(function_));

  std::optional<tensorflow_federated::OwnedValueId> optional_arg_handle;
  if (argument_.has_value()) {
    FCP_ASSIGN_OR_RETURN(
        tensorflow_federated::v0::Value replaced_arg,
        fcp::confidential_compute::ReplaceDatas(
            *argument_,
            [this](std::string uri) { return this->FetchData(uri); },
            [this](std::string uri, std::string key) {
              return this->FetchClientData(uri, key);
            }))
        .replaced_value;
    FCP_ASSIGN_OR_RETURN(
        std::shared_ptr<tensorflow_federated::OwnedValueId> arg_handle,
        fcp::confidential_compute::Embed(replaced_arg, child_executor_));
    optional_arg_handle = std::move(*arg_handle);
  }

  FCP_ASSIGN_OR_RETURN(
      tensorflow_federated::OwnedValueId call_handle,
      child_executor_->CreateCall(fn_handle, optional_arg_handle));
  tensorflow_federated::v0::Value call_result;
  FCP_RETURN_IF_ERROR(child_executor_->Materialize(call_handle, &call_result));
  std::string unencrypted_result = call_result.SerializeAsString();

  // If all inputs are unencrypted, output result can be unencrypted.
  if (input_metadata.has_unencrypted()) {
    BlobMetadata result_metadata;
    result_metadata.set_total_size_bytes(unencrypted_result.size());
    result_metadata.mutable_unencrypted();
    result_metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
    SessionResponse unencrypted_response;
    ReadResponse* unencrypted_read_response =
        unencrypted_response.mutable_read();
    unencrypted_read_response->set_finish_read(true);
    *(unencrypted_read_response->mutable_data()) =
        std::move(unencrypted_result);
    *(unencrypted_read_response->mutable_first_response_metadata()) =
        std::move(result_metadata);
    return unencrypted_response;
  }

  RecordEncryptor encryptor;
  BlobHeader previous_header;
  if (!previous_header.ParseFromString(
          input_metadata.hpke_plus_aead_data().ciphertext_associated_data())) {
    return absl::InvalidArgumentError(
        "Failed to parse the BlobHeader when trying to encrypt outputs.");
  }
  FCP_ASSIGN_OR_RETURN(
      Record result_record,
      encryptor.EncryptRecord(unencrypted_result,
                              input_metadata.hpke_plus_aead_data()
                                  .rewrapped_symmetric_key_associated_data()
                                  .reencryption_public_key(),
                              previous_header.access_policy_sha256(),
                              output_access_policy_node_id_));
  SessionResponse response;
  ReadResponse* read_response = response.mutable_read();
  read_response->set_finish_read(true);
  *(read_response->mutable_data()) =
      std::move(result_record.hpke_plus_aead_data().ciphertext());
  *(read_response->mutable_first_response_metadata()) =
      GetBlobMetadataFromRecord(result_record);
  return response;
}
}  // namespace confidential_federated_compute::tff_server
