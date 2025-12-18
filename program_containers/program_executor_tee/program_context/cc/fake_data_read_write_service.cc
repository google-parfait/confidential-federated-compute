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

#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/crypto_test_utils.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/sync_stream.h"
#include "openssl/rand.h"
#include "program_executor_tee/program_context/cc/kms_helper.h"

namespace confidential_federated_compute::program_executor_tee {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::ReleaseToken;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
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
  std::vector<WriteRequest> requests;
  WriteRequest request;
  while (request_reader->Read(&request)) {
    requests.push_back(request);
  }
  if (requests.size() > 1) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Chunked WriteRequests are not supported by the "
                        "FakeDataReadWriteService");
  }
  BlobMetadata metadata = requests[0].first_request_metadata();
  auto ciphertext = requests[0].data();
  BlobHeader blob_header;
  blob_header.ParseFromString(metadata.hpke_plus_aead_data()
                                  .kms_symmetric_key_associated_data()
                                  .record_header());
  absl::StatusOr<std::string> plaintext_message = message_decryptor_.Decrypt(
      ciphertext, metadata.hpke_plus_aead_data().ciphertext_associated_data(),
      metadata.hpke_plus_aead_data().encrypted_symmetric_key(),
      metadata.hpke_plus_aead_data()
          .kms_symmetric_key_associated_data()
          .record_header(),
      metadata.hpke_plus_aead_data().encapsulated_public_key(),
      blob_header.key_id());
  if (!plaintext_message.ok()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Decryption failed: " +
                            std::string(plaintext_message.status().message()));
  }
  released_data_[requests[0].key()] = *plaintext_message;

  absl::StatusOr<ReleaseToken> token =
      ReleaseToken::Decode(requests[0].release_token());
  if (!token.ok()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "Decoding release token failed: " +
                            std::string(token.status().message()));
  }
  released_state_changes_[requests[0].key()] = {token->src_state,
                                                token->dst_state};
  return grpc::Status::OK;
}

absl::Status FakeDataReadWriteService::StoreEncryptedMessageForKms(
    absl::string_view uri, absl::string_view message,
    std::optional<absl::string_view> blob_id) {
  if (uri_to_read_response_.find(std::string(uri)) !=
      uri_to_read_response_.end()) {
    return absl::InvalidArgumentError("Uri already set.");
  }

  BlobHeader header;
  if (blob_id.has_value()) {
    header.set_blob_id(std::string(*blob_id));
  } else {
    std::string random_blob_id(kBlobIdSize, '\0');
    (void)RAND_bytes(reinterpret_cast<unsigned char*>(random_blob_id.data()),
                     random_blob_id.size());
    header.set_blob_id(random_blob_id);
  }
  header.set_key_id(kInputKeyId);
  header.set_access_policy_sha256(kAccessPolicyHash);
  std::string associated_data = header.SerializeAsString();

  MessageEncryptor encryptor;
  FCP_ASSIGN_OR_RETURN(
      EncryptMessageResult encrypt_result,
      encryptor.Encrypt(message, input_public_private_key_pair_.first,
                        associated_data));

  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(encrypt_result.ciphertext.size());
  BlobMetadata::HpkePlusAeadMetadata* encryption_metadata =
      metadata.mutable_hpke_plus_aead_data();
  encryption_metadata->set_ciphertext_associated_data(associated_data);
  encryption_metadata->set_encrypted_symmetric_key(
      encrypt_result.encrypted_symmetric_key);
  encryption_metadata->set_encapsulated_public_key(encrypt_result.encapped_key);
  encryption_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header(associated_data);

  ReadResponse response;
  *response.mutable_first_response_metadata() = std::move(metadata);
  *response.mutable_data() = std::move(encrypt_result.ciphertext);
  response.set_finish_read(true);

  uri_to_read_response_[std::string(uri)] = std::move(response);
  return absl::OkStatus();
}

absl::Status FakeDataReadWriteService::StorePlaintextMessage(
    absl::string_view uri, absl::string_view message) {
  if (uri_to_read_response_.find(std::string(uri)) !=
      uri_to_read_response_.end()) {
    return absl::InvalidArgumentError("Uri already set.");
  }

  BlobMetadata blob_metadata;
  blob_metadata.set_total_size_bytes(message.size());
  blob_metadata.mutable_unencrypted();
  blob_metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);

  ReadResponse response;
  *response.mutable_first_response_metadata() = std::move(blob_metadata);
  response.set_finish_read(true);
  response.set_data(std::string(message));

  uri_to_read_response_[std::string(uri)] = std::move(response);
  return absl::OkStatus();
}

}  // namespace confidential_federated_compute::program_executor_tee