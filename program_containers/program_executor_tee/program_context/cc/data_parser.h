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
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

// Increase gRPC message size limit to 2GB
inline constexpr int kMaxGrpcMessageSize = 2 * 1000 * 1000 * 1000;

class DataParser {
 public:
  DataParser(
      confidential_federated_compute::BlobDecryptor* blob_decryptor,
      std::string outgoing_server_address, bool use_caching = true,
      std::function<std::string()> nonce_generator = []() {
        return fcp::RandomToken::Generate().ToString();
      });

  // Retrieves the TensorProto that is described by the provided uri and
  // FcCheckpoint key.
  absl::StatusOr<tensorflow_federated::aggregation::TensorProto>
  ResolveUriToTensor(std::string uri, std::string key);

 private:
  // Retrieve the FC checkpoint for a uri, either by using the cache or
  // sending a ReadRequest to the DataReadWrite service.
  absl::StatusOr<std::string> ResolveUriToFcCheckpoint(std::string uri);

  // Parse a ReadResponse message into a FC checkpoint, decrypting it and
  // checking that the nonce matches, if necessary.
  absl::StatusOr<std::string> ParseReadResponseToFcCheckpoint(
      const fcp::confidentialcompute::outgoing::ReadResponse& read_response,
      const std::string& nonce);

  BlobDecryptor* blob_decryptor_;
  std::unique_ptr<
      fcp::confidentialcompute::outgoing::DataReadWrite::StubInterface>
      stub_;
  bool use_caching_;
  absl::Mutex cache_mutex_;
  absl::flat_hash_map<std::string, std::string> uri_to_checkpoint_cache_;
  std::function<std::string()> nonce_generator_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_PARSER_H_