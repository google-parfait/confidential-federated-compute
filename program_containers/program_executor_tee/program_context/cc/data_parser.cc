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

using ::confidential_federated_compute::BlobDecryptor;
using ::fcp::confidentialcompute::BlobHeader;
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
    confidential_federated_compute::BlobDecryptor* blob_decryptor,
    std::string outgoing_server_address, bool use_kms,
    std::string reencryption_key, std::string reencryption_policy_hash,
    PrivateState* private_state,
    std::shared_ptr<oak::crypto::SigningKeyHandle> signing_key_handle,
    std::set<std::string> authorized_logical_pipeline_policies_hashes,
    std::function<std::string()> nonce_generator)
    : blob_decryptor_(blob_decryptor),
      use_kms_(use_kms),
      private_state_(private_state),
      signing_key_handle_(signing_key_handle),
      nonce_generator_(nonce_generator) {
  // All hashes and encryption keys are Base64Escaped before being passed over
  // the pybind boundary, so they must be decoded here.
  if (use_kms_) {
    std::string decoded_reencryption_key;
    absl::Base64Unescape(reencryption_key, &decoded_reencryption_key);
    reencryption_key_ = decoded_reencryption_key;

    std::string decoded_reencryption_policy_hash;
    absl::Base64Unescape(reencryption_policy_hash,
                         &decoded_reencryption_policy_hash);
    reencryption_policy_hash_ = reencryption_policy_hash;

    for (const auto& hash : authorized_logical_pipeline_policies_hashes) {
      std::string decoded_hash;
      absl::Base64Unescape(hash, &decoded_hash);
      authorized_logical_pipeline_policies_hashes_.insert(decoded_hash);
    }
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
  // Check whether the uri is already in the cache if the ledger is being used.
  if (!use_kms_) {
    absl::MutexLock lock(&cache_mutex_);
    auto it = uri_to_checkpoint_cache_.find(uri);
    if (it != uri_to_checkpoint_cache_.end()) {
      return it->second;
    }
  }

  // Make a ReadRequest for the uri and reconstruct a combined ReadResponse.
  // Only use a nonce if the ledger is being used.
  ClientContext client_context;
  ReadRequest read_request;
  read_request.set_uri(uri);
  std::optional<std::string> nonce = std::nullopt;
  if (!use_kms_) {
    nonce = nonce_generator_();
    read_request.set_nonce(*nonce);
  }
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

  // Add the checkpoint to the cache if the ledger is being used.
  FCP_ASSIGN_OR_RETURN(
      std::string checkpoint,
      ParseReadResponseToFcCheckpoint(uri, combined_read_response, nonce));
  if (!use_kms_) {
    absl::MutexLock lock(&cache_mutex_);
    uri_to_checkpoint_cache_[uri] = checkpoint;
  }
  return checkpoint;
}

absl::StatusOr<std::string> DataParser::ParseReadResponseToFcCheckpoint(
    std::string uri, const ReadResponse& read_response,
    const std::optional<std::string>& nonce) {
  if (read_response.first_response_metadata().has_unencrypted()) {
    return read_response.data();
  }

  // Check the nonce if the ledger is being used.
  if (!use_kms_) {
    if (read_response.first_response_metadata()
            .hpke_plus_aead_data()
            .rewrapped_symmetric_key_associated_data()
            .nonce() != *nonce) {
      return absl::InvalidArgumentError("ReadResponse nonce does not match");
    }
    return blob_decryptor_->DecryptBlob(read_response.first_response_metadata(),
                                        read_response.data());
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

absl::Status DataParser::ReleaseUnencrypted(std::string data, std::string key) {
  WriteRequest write_request;
  std::string next_state = private_state_->GetReleaseUpdateState();
  FCP_RETURN_IF_ERROR(CreateWriteRequestForRelease(
      &write_request, *signing_key_handle_, reencryption_key_, key, data,
      reencryption_policy_hash_, private_state_->GetReleaseInitialState(),
      next_state));

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
  private_state_->SetReleaseInitialState(next_state);
  return absl::OkStatus();
}

}  // namespace confidential_federated_compute::program_executor_tee