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

#include "containers/program_executor_tee/program_context/cc/data_uri_resolver.h"

#include <future>
#include <map>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "containers/crypto.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "fcp/protos/confidentialcompute/file_info.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "grpcpp/client_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/converters.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

using ::fcp::confidentialcompute::outgoing::DataReadWrite;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;

namespace {

// Extract the tensor stored at the provided checkpoint key into a tff Value.
absl::StatusOr<tensorflow_federated::v0::Value> GetTensorValueFromCheckpoint(
    absl::string_view checkpoint, absl::string_view key) {
  FederatedComputeCheckpointParserFactory parser_factory;
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<CheckpointParser> parser,
                       parser_factory.Create(absl::Cord(checkpoint)));
  FCP_ASSIGN_OR_RETURN(tensorflow_federated::aggregation::Tensor agg_tensor,
                       parser->GetTensor(std::string(key)));
  FCP_ASSIGN_OR_RETURN(
      tensorflow::Tensor tensor,
      tensorflow_federated::aggregation::tensorflow::ToTfTensor(
          std::move(agg_tensor)));
  tensorflow_federated::v0::Value resolved_value;
  FCP_RETURN_IF_ERROR(tensorflow_federated::SerializeTensorValue(
      std::move(tensor), &resolved_value));
  return resolved_value;
}

}  // namespace

absl::Status DataUriResolver::ResolveToValue(
    const federated_language::Data& data_reference,
    const federated_language::Type& data_type,
    tensorflow_federated::v0::Value& value_out) {
  fcp::confidentialcompute::FileInfo file_info;
  if (!data_reference.content().UnpackTo(&file_info)) {
    return absl::InvalidArgumentError(
        "Expected Data content field to contain FileInfo proto.");
  }

  // Only send a ReadRequest for the uri if we have not already sent one.
  // If we are going to send a ReadRequest, immediately populate the
  // uri_to_value_cache_ with a promise corresponding to this uri so that
  // other calls to this function for the same uri will not also trigger
  // a ReadRequest.
  std::promise<std::string> promise;
  bool send_request = false;
  {
    absl::MutexLock l(&mutex_);
    if (uri_to_value_cache_.find(file_info.uri()) ==
        uri_to_value_cache_.end()) {
      uri_to_value_cache_[file_info.uri()] = promise.get_future().share();
      send_request = true;
    }
  }

  // Send a ReadRequest, if needed, and update the promise to hold the
  // decrypted result, which will be a federated compute checkpoint.
  if (send_request) {
    FCP_ASSIGN_OR_RETURN(std::string unencrypted_data,
                         ResolveUriToCheckpoint(file_info.uri()));
    promise.set_value(std::move(unencrypted_data));
  }

  // Set the output value to a tff Value representing the tensor at the given
  // key in the federated compute checkpoint that was retrieved for this uri.
  FCP_ASSIGN_OR_RETURN(
      value_out,
      GetTensorValueFromCheckpoint(uri_to_value_cache_[file_info.uri()].get(),
                                   file_info.key()));

  return absl::OkStatus();
}

absl::StatusOr<std::string> DataUriResolver::ResolveUriToCheckpoint(
    absl::string_view uri) {
  fcp::confidentialcompute::outgoing::ReadRequest read_request;
  read_request.set_uri(std::string(uri));
  std::string nonce = nonce_generator_();
  read_request.set_nonce(nonce);

  grpc::ClientContext client_context;
  std::unique_ptr<
      grpc::ClientReader<fcp::confidentialcompute::outgoing::ReadResponse>>
      reader(stub_->Read(&client_context, read_request));

  // For now we assume the data returned in the ReadResponse is not chunked.
  fcp::confidentialcompute::outgoing::ReadResponse read_response;
  while (reader->Read(&read_response)) {
    if (!read_response.has_first_response_metadata()) {
      return absl::InternalError("Expecting only one ReadResponse.");
    }

    // If the metadata indicates that the data is unencrypted, just return it.
    if (read_response.first_response_metadata().has_unencrypted()) {
      return read_response.data();
    }

    // Otherwise, check that the nonce in the ReadResponse matches the
    // ReadRequest and decrypt the data.
    if (read_response.first_response_metadata()
            .hpke_plus_aead_data()
            .rewrapped_symmetric_key_associated_data()
            .nonce() != nonce) {
      return absl::InternalError("Mismatched nonce.");
    }
    absl::StatusOr<std::string> decrypted_data = blob_decryptor_->DecryptBlob(
        read_response.first_response_metadata(), read_response.data());
    return decrypted_data;
  }
  return absl::InternalError("Error receiving ReadResponse.");
}

}  // namespace confidential_federated_compute::program_executor_tee