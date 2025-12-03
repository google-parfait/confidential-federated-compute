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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_FAKE_DATA_READ_WRITE_SERVICE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_FAKE_DATA_READ_WRITE_SERVICE_H_

#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "containers/crypto_test_utils.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/sync_stream.h"

namespace confidential_federated_compute::program_executor_tee {

constexpr char kInputKeyId[] = "input";
constexpr char kAccessPolicyHash[] = "access_policy_hash";
constexpr size_t kBlobIdSize = 16;

// Fake DataReadWrite service that retains information about past requests
// and can send mock responses.
class FakeDataReadWriteService
    : public fcp::confidentialcompute::outgoing::DataReadWrite::Service {
 public:
  FakeDataReadWriteService(bool use_kms = false)
      : use_kms_(use_kms),
        input_public_private_key_pair_(
            crypto_test_utils::GenerateKeyPair("input")),
        result_public_private_key_pair_(
            crypto_test_utils::GenerateKeyPair("result")),
        message_decryptor_(
            /*config_properties=*/"",
            std::vector<absl::string_view>(
                {result_public_private_key_pair_.second})) {}

  grpc::Status Read(
      ::grpc::ServerContext*,
      const ::fcp::confidentialcompute::outgoing::ReadRequest* request,
      grpc::ServerWriter<::fcp::confidentialcompute::outgoing::ReadResponse>*
          response_writer) override;

  grpc::Status Write(
      ::grpc::ServerContext*,
      ::grpc::ServerReader<fcp::confidentialcompute::outgoing::WriteRequest>*
          request_reader,
      fcp::confidentialcompute::outgoing::WriteResponse*) override;

  // Encrypts the provided message and stores it within a ReadResponse in an
  // internal map such that the ReadResponse will be returned in response to a
  // later ReadRequest for the given uri.
  // TODO: b/451714072 - This method is specific to the ledger and will be
  // deleted once the KMS migration is complete. The method below will persist.
  absl::Status StoreEncryptedMessageForLedger(
      absl::string_view uri, absl::string_view message,
      absl::string_view ciphertext_associated_data,
      absl::string_view recipient_public_key, absl::string_view nonce,
      absl::string_view reencryption_public_key);

  // Encrypts the provided message and stores it within a ReadResponse in an
  // internal map such that the ReadResponse will be returned in response to a
  // later ReadRequest for the given uri.
  absl::Status StoreEncryptedMessageForKms(absl::string_view uri,
                                           absl::string_view message);

  // Stores the provided message within a ReadResponse in an internal map such
  // that the ReadResponse will be returned in response to a later ReadRequest
  // for the given uri.
  absl::Status StorePlaintextMessage(absl::string_view uri,
                                     absl::string_view message);

  std::pair<std::string, std::string> GetInputPublicPrivateKeyPair() {
    return input_public_private_key_pair_;
  }

  std::pair<std::string, std::string> GetResultPublicPrivateKeyPair() {
    return result_public_private_key_pair_;
  }

  // Returns a list of uris from received ReadRequest args.
  std::vector<std::string> GetReadRequestUris() { return read_request_uris_; }

  // Returns a map of key to plaintext messages received by the Write endpoint.
  std::map<std::string, std::string> GetReleasedData() {
    return released_data_;
  }

 private:
  // Whether or not KMS is being used.
  // TODO: b/451714072 - Delete this once the KMS migration is complete.
  bool use_kms_;

  // The keypair to use when storing uploads in the KMS case.
  std::pair<std::string, std::string> input_public_private_key_pair_;

  // The keypair to use when releasing results in the KMS case.
  std::pair<std::string, std::string> result_public_private_key_pair_;

  // Map that stores the ReadResponse to return for a ReadRequest with a
  // particular uri.
  std::map<std::string, fcp::confidentialcompute::outgoing::ReadResponse>
      uri_to_read_response_;

  // List of uris from received ReadRequest args.
  std::vector<std::string> read_request_uris_;

  // A decryptor that can be used to retrieve the plaintext message from a
  // WriteRequest.
  fcp::confidential_compute::MessageDecryptor message_decryptor_;

  // A map of key to plaintext messages received by the Write endpoint.
  std::map<std::string, std::string> released_data_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_FAKE_DATA_READ_WRITE_SERVICE_H_