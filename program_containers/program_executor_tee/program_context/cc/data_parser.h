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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_PARSER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_PARSER_H_

#include <functional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "containers/crypto.h"
#include "fcp/base/random_token.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "program_executor_tee/private_state.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

// Increase gRPC message size limit to 2GB
inline constexpr int kMaxGrpcMessageSize = 2 * 1000 * 1000 * 1000;

class DataParser {
 public:
  DataParser(
      confidential_federated_compute::BlobDecryptor* blob_decryptor,
      std::string outgoing_server_address, std::string reencryption_key = "",
      std::string reencryption_policy_hash = "",
      PrivateState* private_state = nullptr,
      std::shared_ptr<oak::crypto::SigningKeyHandle> signing_key_handle =
          nullptr,
      std::set<std::string> authorized_logical_pipeline_policies_hashes = {});

  // Retrieves the TensorProto that is described by the provided uri and
  // FcCheckpoint key.
  absl::StatusOr<tensorflow_federated::aggregation::TensorProto>
  ResolveUriToTensor(std::string uri, std::string key);

  // Wraps the data in a release token and sends it to untrusted space.
  absl::Status ReleaseUnencrypted(std::string data, std::string key);

 private:
  // Retrieve the FC checkpoint for a uri, either by using the cache or
  // sending a ReadRequest to the DataReadWrite service.
  absl::StatusOr<std::string> ResolveUriToFcCheckpoint(std::string uri);

  // Parse a ReadResponse message into a FC checkpoint.
  absl::StatusOr<std::string> ParseReadResponseToFcCheckpoint(
      std::string uri,
      const fcp::confidentialcompute::outgoing::ReadResponse& read_response);

  BlobDecryptor* blob_decryptor_;
  std::unique_ptr<
      fcp::confidentialcompute::outgoing::DataReadWrite::StubInterface>
      stub_;

  // The reencryption keys used to re-encrypt the final blobs.
  std::string reencryption_key_;
  // The policy hash used to re-encrypt the final blobs with.
  std::string reencryption_policy_hash_;
  // Private state.
  PrivateState* private_state_;
  // The signing key handle used to sign the final result.
  std::shared_ptr<oak::crypto::SigningKeyHandle> signing_key_handle_;
  // The authorized logical policy hashes for this container.
  std::set<std::string> authorized_logical_pipeline_policies_hashes_;

  // To prevent orchestrator attacks where the same blob is returned for
  // multiple filenames, we maintain a blob_id -> filename map and check that a
  // given blob_id is only seen for one filename.
  absl::flat_hash_map<std::string, std::string> blob_id_to_filename_map_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_PARSER_H_