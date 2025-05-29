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

#include "containers/program_executor_tee/program_context/cc/fake_data_read_write_service.h"

#include <map>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/blob_metadata.h"
#include "containers/crypto_test_utils.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/sync_stream.h"

namespace confidential_federated_compute::program_executor_tee {

using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::outgoing::ReadRequest;
using ::fcp::confidentialcompute::outgoing::ReadResponse;
using ::fcp::confidentialcompute::outgoing::WriteRequest;
using ::fcp::confidentialcompute::outgoing::WriteResponse;

grpc::Status FakeDataReadWriteService::Read(
    ::grpc::ServerContext*, const ReadRequest* request,
    grpc::ServerWriter<ReadResponse>* response_writer) {
  if (uri_to_read_response_.find(request->uri()) ==
      uri_to_read_response_.end()) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND,
                        "Requested uri " + request->uri() + " not found.");
  }
  response_writer->Write(uri_to_read_response_[request->uri()]);
  // Store the uri from the request.
  read_request_uris_.push_back(request->uri());
  return grpc::Status::OK;
}

grpc::Status FakeDataReadWriteService::Write(
    ::grpc::ServerContext*, ::grpc::ServerReader<WriteRequest>* request_reader,
    WriteResponse*) {
  // Append the stream of requests to write_call_args_.
  std::vector<WriteRequest> requests;
  WriteRequest request;
  while (request_reader->Read(&request)) {
    requests.push_back(request);
  }
  write_call_args_.push_back(requests);
  return grpc::Status::OK;
}

absl::Status FakeDataReadWriteService::StoreEncryptedMessage(
    absl::string_view uri, absl::string_view message,
    absl::string_view ciphertext_associated_data,
    absl::string_view recipient_public_key, absl::string_view nonce,
    absl::string_view reencryption_public_key) {
  if (uri_to_read_response_.find(std::string(uri)) !=
      uri_to_read_response_.end()) {
    return absl::InvalidArgumentError("Uri already set.");
  }

  FCP_ASSIGN_OR_RETURN(
      absl::StatusOr<Record> rewrapped_record,
      crypto_test_utils::CreateRewrappedRecord(
          message, ciphertext_associated_data, recipient_public_key, nonce,
          reencryption_public_key));

  ReadResponse response;
  *response.mutable_first_response_metadata() =
      GetBlobMetadataFromRecord(*rewrapped_record);
  response.set_finish_read(true);
  response.set_data(rewrapped_record->hpke_plus_aead_data().ciphertext());

  uri_to_read_response_[std::string(uri)] = std::move(response);
  return absl::OkStatus();
}

absl::Status FakeDataReadWriteService::StorePlaintextMessage(
    absl::string_view uri, absl::string_view message) {
  if (uri_to_read_response_.find(std::string(uri)) !=
      uri_to_read_response_.end()) {
    return absl::InvalidArgumentError("Uri already set.");
  }

  Record record;
  record.set_unencrypted_data(std::string(message));
  record.set_compression_type(Record::COMPRESSION_TYPE_NONE);

  ReadResponse response;
  *response.mutable_first_response_metadata() =
      GetBlobMetadataFromRecord(record);
  response.set_finish_read(true);
  response.set_data(record.unencrypted_data());

  uri_to_read_response_[std::string(uri)] = std::move(response);
  return absl::OkStatus();
}

std::vector<std::string> FakeDataReadWriteService::GetReadRequestUris() {
  return read_request_uris_;
}

std::vector<std::vector<WriteRequest>>
FakeDataReadWriteService::GetWriteCallArgs() {
  return write_call_args_;
}

}  // namespace confidential_federated_compute::program_executor_tee