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
#include "containers/crypto.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::program_executor_tee {

using ::confidential_federated_compute::BlobDecryptor;
using ::fcp::confidentialcompute::outgoing::ReadRequest;
using ::fcp::confidentialcompute::outgoing::ReadResponse;
using ::grpc::ClientContext;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorProto;

DataParser::DataParser(
    confidential_federated_compute::BlobDecryptor* blob_decryptor,
    std::string outgoing_server_address, bool use_caching,
    std::function<std::string()> nonce_generator)
    : blob_decryptor_(blob_decryptor),
      use_caching_(use_caching),
      nonce_generator_(nonce_generator) {
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
  // Check whether the uri is already in the cache.
  if (use_caching_) {
    absl::MutexLock lock(&cache_mutex_);
    auto it = uri_to_checkpoint_cache_.find(uri);
    if (it != uri_to_checkpoint_cache_.end()) {
      return it->second;
    }
  }

  // Make a ReadRequest for the uri and reconstruct a combined ReadResponse.
  ClientContext client_context;
  ReadRequest read_request;
  read_request.set_uri(uri);
  std::string nonce = nonce_generator_();
  read_request.set_nonce(nonce);
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

  // Add the checkpoint to the cache.
  FCP_ASSIGN_OR_RETURN(
      std::string checkpoint,
      ParseReadResponseToFcCheckpoint(combined_read_response, nonce));
  if (use_caching_) {
    absl::MutexLock lock(&cache_mutex_);
    uri_to_checkpoint_cache_[uri] = checkpoint;
  }
  return checkpoint;
}

absl::StatusOr<std::string> DataParser::ParseReadResponseToFcCheckpoint(
    const ReadResponse& read_response, const std::string& nonce) {
  if (read_response.first_response_metadata().has_unencrypted()) {
    return read_response.data();
  }
  if (read_response.first_response_metadata()
          .hpke_plus_aead_data()
          .rewrapped_symmetric_key_associated_data()
          .nonce() != nonce) {
    return absl::InvalidArgumentError("ReadResponse nonce does not match");
  }
  return blob_decryptor_->DecryptBlob(read_response.first_response_metadata(),
                                      read_response.data());
}

}  // namespace confidential_federated_compute::program_executor_tee