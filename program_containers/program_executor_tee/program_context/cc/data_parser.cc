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

#include "program_executor_tee/program_context/cc/data_parser.h"

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "cc/crypto/signing_key.h"
#include "containers/crypto.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "program_executor_tee/private_state.h"
#include "program_executor_tee/program_context/cc/kms_helper.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::program_executor_tee {

using ::confidential_federated_compute::Decryptor;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::outgoing::IntermediateResult;
using ::fcp::confidentialcompute::outgoing::ReadRequest;
using ::fcp::confidentialcompute::outgoing::ReadResponse;
using ::fcp::confidentialcompute::outgoing::WriteRequest;
using ::fcp::confidentialcompute::outgoing::WriteResponse;
using ::grpc::ClientContext;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorProto;

DataParser::DataParser(
    confidential_federated_compute::Decryptor* blob_decryptor,
    std::string outgoing_server_address,
    std::vector<std::string> reencryption_keys,
    std::string reencryption_policy_hash, std::string kms_public_key,
    std::string invocation_id, PrivateState* private_state,
    std::shared_ptr<oak::crypto::SigningKeyHandle> signing_key_handle,
    std::set<std::string> authorized_logical_pipeline_policies_hashes)
    : blob_decryptor_(blob_decryptor),
      invocation_id_(invocation_id),
      private_state_(private_state),
      signing_key_handle_(signing_key_handle) {
  absl::Base64Unescape(kms_public_key, &kms_public_key_);
  // All hashes and encryption keys are Base64Escaped before being passed over
  // the pybind boundary, so they must be decoded here.
  for (const auto& reencryption_key : reencryption_keys) {
    std::string decoded_reencryption_key;
    absl::Base64Unescape(reencryption_key, &decoded_reencryption_key);
    reencryption_keys_.push_back(decoded_reencryption_key);
  }

  std::string decoded_reencryption_policy_hash;
  absl::Base64Unescape(reencryption_policy_hash,
                       &decoded_reencryption_policy_hash);
  reencryption_policy_hash_ = decoded_reencryption_policy_hash;

  for (const auto& hash : authorized_logical_pipeline_policies_hashes) {
    std::string decoded_hash;
    absl::Base64Unescape(hash, &decoded_hash);
    authorized_logical_pipeline_policies_hashes_.insert(decoded_hash);
  }

  grpc::ChannelArguments args;
  args.SetMaxSendMessageSize(kMaxGrpcMessageSize);
  args.SetMaxReceiveMessageSize(kMaxGrpcMessageSize);
  stub_ = fcp::confidentialcompute::outgoing::DataReadWrite::NewStub(
      grpc::CreateCustomChannel(outgoing_server_address,
                                grpc::InsecureChannelCredentials(), args));
}

absl::StatusOr<TensorProto> DataParser::ResolveUriToTensor(std::string uri,
                                                           std::string key) {
  FCP_ASSIGN_OR_RETURN(std::string fc_checkpoint,
                       ResolveUriToFcCheckpoint(uri));
  FederatedComputeCheckpointParserFactory parser_factory;
  FCP_ASSIGN_OR_RETURN(
      std::unique_ptr<CheckpointParser> parser,
      parser_factory.Create(absl::Cord(std::move(fc_checkpoint))));
  FCP_ASSIGN_OR_RETURN(tensorflow_federated::aggregation::Tensor agg_tensor,
                       parser->GetTensor(key));
  return agg_tensor.ToProto();
}

absl::StatusOr<std::string> DataParser::ResolveUriToFcCheckpoint(
    std::string uri) {
  ClientContext client_context;
  ReadRequest read_request;
  read_request.set_uri(uri);
  auto reader = stub_->Read(&client_context, read_request);
  ReadResponse combined_read_response;
  std::string combined_data = "";
  ReadResponse response;
  while (reader->Read(&response)) {
    if (response.has_first_response_metadata()) {
      *combined_read_response.mutable_first_response_metadata() =
          std::move(response.first_response_metadata());
    }
    combined_data += response.data();
    if (response.finish_read()) {
      combined_read_response.set_data(combined_data);
      combined_read_response.set_finish_read(true);
    }
  }
  FCP_RETURN_IF_ERROR(fcp::base::FromGrpcStatus(reader->Finish()));
  return ParseReadResponseToFcCheckpoint(uri, combined_read_response);
}

absl::StatusOr<std::string> DataParser::ParseReadResponseToFcCheckpoint(
    std::string uri, const ReadResponse& read_response) {
  if (read_response.first_response_metadata().has_unencrypted()) {
    return read_response.data();
  }

  // Parse the BlobHeader to get the access policy hash and key ID.
  BlobHeader blob_header;
  if (!blob_header.ParseFromString(read_response.first_response_metadata()
                                       .hpke_plus_aead_data()
                                       .kms_symmetric_key_associated_data()
                                       .record_header())) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data.record_header() cannot be "
        "parsed to BlobHeader.");
  }

  // Verify that the access policy hash matches one of the authorized
  // logical pipeline policy hashes returned by KMS before returning
  // the key ID.
  if (authorized_logical_pipeline_policies_hashes_.find(
          blob_header.access_policy_sha256()) ==
      authorized_logical_pipeline_policies_hashes_.end()) {
    return absl::InvalidArgumentError(
        "BlobHeader.access_policy_sha256 does not match any "
        "authorized_logical_pipeline_policies_hashes returned by "
        "KMS.");
  }

  // Check that the returned blob has a blob id that has either never been seen
  // before, or was previously seen for the same filename. A malicious
  // orchestrator could return the same blob for multiple filenames.
  if (blob_id_to_filename_map_.find(blob_header.blob_id()) ==
      blob_id_to_filename_map_.end()) {
    blob_id_to_filename_map_[blob_header.blob_id()] = uri;
  } else if (blob_id_to_filename_map_[blob_header.blob_id()] != uri) {
    return absl::InvalidArgumentError(
        "This blob id was previously returned for a different filename.");
  }

  return blob_decryptor_->DecryptBlob(read_response.first_response_metadata(),
                                      read_response.data(),
                                      blob_header.key_id());
}

absl::Status DataParser::ReleaseUnencryptedInternal(std::string data,
                                                    std::string key) {
  WriteRequest write_request;
  FCP_RETURN_IF_ERROR(CreateWriteRequestForRelease(
      &write_request, *signing_key_handle_,
      reencryption_keys_[kReleaseValueEncryptionKeyIndex], key, data,
      reencryption_policy_hash_, private_state_->GetState(),
      private_state_->CommitNewState()));

  ClientContext client_context;
  WriteResponse response;
  std::unique_ptr<::grpc::ClientWriterInterface<WriteRequest>> writer =
      stub_->Write(&client_context, &response);

  if (!writer->Write(write_request)) {
    return absl::InternalError("Failed to write WriteRequest");
  }
  if (!writer->WritesDone() || !writer->Finish().ok()) {
    return absl::InternalError("Failed to complete Write");
  }
  return absl::OkStatus();
}

absl::Status DataParser::ReleaseUnencrypted(std::string data, std::string key) {
  if (private_state_->HasPriorSaveRecovery()) {
    return absl::FailedPreconditionError(
        "Releasing unencrypted information without associated recovery "
        "information is unsupported if there has been a prior attempt to "
        "save recovery information.");
  }
  return ReleaseUnencryptedInternal(std::move(data), std::move(key));
}

absl::Status DataParser::SaveRecoveryInfo(
    std::string recovery_value, std::string recovery_key,
    std::vector<std::pair<std::string, std::string>> release_queue) {
  // Build the RecoveryInfo proto from the raw value and the committed recovery
  // blob ID from the private state.
  RecoveryInfo recovery_info;
  recovery_info.set_value(std::move(recovery_value));
  auto committed_blob_id = private_state_->GetCommittedRecoveryBlobId();
  if (committed_blob_id.has_value()) {
    recovery_info.set_committed_blob_id(*committed_blob_id);
  }

  WriteRequest write_request;
  std::string blob_id;
  FCP_RETURN_IF_ERROR(CreateWriteRequestForEncryptedValue(
      &write_request, &blob_id, *signing_key_handle_,
      reencryption_keys_[kRecoveryInfoEncryptionKeyIndex], recovery_key,
      recovery_info.SerializeAsString(), reencryption_policy_hash_));
  ClientContext client_context;
  WriteResponse response;
  std::unique_ptr<::grpc::ClientWriterInterface<WriteRequest>> writer =
      stub_->Write(&client_context, &response);

  if (!writer->Write(write_request)) {
    return absl::InternalError("Failed to write WriteRequest");
  }
  if (!writer->WritesDone() || !writer->Finish().ok()) {
    return absl::InternalError("Failed to complete Write");
  }

  // Update the private state with the new recovery blob id. Next time an
  // unencrypted value is released, the updated private state will be committed
  // to KMS and recovery from earlier RecoveryInfo messages will become
  // unallowed.
  private_state_->SetRecoveryBlobId(blob_id);

  for (const auto& [data, key] : release_queue) {
    FCP_RETURN_IF_ERROR(ReleaseUnencryptedInternal(data, key));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::string> DataParser::RestoreRecoveryInfo(
    std::string recovery_key) {
  ReadRequest read_request;
  read_request.set_uri(recovery_key);
  ClientContext client_context;
  auto reader = stub_->Read(&client_context, read_request);
  ReadResponse response;
  std::string combined_data = "";
  fcp::confidentialcompute::BlobMetadata first_response_metadata;
  bool is_first = true;
  while (reader->Read(&response)) {
    if (is_first) {
      first_response_metadata = response.first_response_metadata();
      is_first = false;
    }
    combined_data += response.data();
    if (response.finish_read()) {
      break;
    }
  }
  FCP_RETURN_IF_ERROR(fcp::base::FromGrpcStatus(reader->Finish()));

  // Parse the BlobHeader to get the access policy hash and key ID.
  BlobHeader blob_header;
  if (!blob_header.ParseFromString(first_response_metadata.hpke_plus_aead_data()
                                       .kms_symmetric_key_associated_data()
                                       .record_header())) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data.record_header() cannot be "
        "parsed to BlobHeader.");
  }

  IntermediateResult intermediate_result;
  if (!intermediate_result.ParseFromString(combined_data)) {
    return absl::InvalidArgumentError(
        "Failed to parse IntermediateResult from combined data.");
  }

  FCP_ASSIGN_OR_RETURN(std::string decrypted_data,
                       blob_decryptor_->DecryptBlob(first_response_metadata,
                                                    intermediate_result.data(),
                                                    blob_header.key_id()));

  FCP_ASSIGN_OR_RETURN(
      BlobProvenance blob_provenance,
      VerifyBlobProvenance(decrypted_data, intermediate_result.signature(),
                           intermediate_result.signing_key_endorsement(),
                           kms_public_key_, invocation_id_));
  if (blob_provenance.transform_index != 0) {
    return absl::InvalidArgumentError(
        "The recovery info blob must be produced by the first transform in the "
        "pipeline.");
  }

  RecoveryInfo recovery_info;
  if (!recovery_info.ParseFromString(decrypted_data)) {
    return absl::InvalidArgumentError(
        "Failed to parse RecoveryInfo from decrypted data.");
  }

  if (!private_state_->AllowRecovery(blob_header.blob_id(), recovery_info)) {
    return absl::InvalidArgumentError(
        "The recovery info does not match the expected state for recovery.");
  }

  return recovery_info.value();
}

}  // namespace confidential_federated_compute::program_executor_tee