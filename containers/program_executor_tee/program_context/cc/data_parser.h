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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_PARSER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_PARSER_H_

#include <string>

#include "absl/status/statusor.h"
#include "containers/crypto.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

class DataParser {
 public:
  DataParser(confidential_federated_compute::BlobDecryptor* blob_decryptor)
      : blob_decryptor_(blob_decryptor) {}

  // Parses a ReadResponse message into a FC checkpoint, extracts the tensor at
  // the provided key, and returns a TFF value proto representing the tensor.
  absl::StatusOr<tensorflow_federated::v0::Value> ParseReadResponseToValue(
      const fcp::confidentialcompute::outgoing::ReadResponse& read_response,
      const std::string& nonce, const std::string& key);

  // Convert an AggCore tensor into a TFF value proto.
  static absl::StatusOr<tensorflow_federated::v0::Value>
  ConvertAggCoreTensorToValue(
      const tensorflow_federated::aggregation::Tensor& tensor);

 private:
  // Parse a ReadResponse message into a FC checkpoint, decrypting it and
  // checking that the nonce matches, if necessary.
  absl::StatusOr<std::string> ParseReadResponseToFcCheckpoint(
      const fcp::confidentialcompute::outgoing::ReadResponse& read_response,
      const std::string& nonce);

  BlobDecryptor* blob_decryptor_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_PARSER_H_