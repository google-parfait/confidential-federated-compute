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
#include <map>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "cc/crypto/signing_key.h"
#include "containers/crypto.h"
#include "fcp/base/random_token.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "program_executor_tee/private_state.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

// Increase gRPC message size limit to 2GB
inline constexpr int kMaxGrpcMessageSize = 2 * 1000 * 1000 * 1000;

// The reencryption key indices used to re-encrypt the final blobs.
static constexpr int kRecoveryInfoEncryptionKeyIndex = 0;
static constexpr int kReleaseValueEncryptionKeyIndex = 1;
static constexpr int kNumEncryptionKeys = 2;

// DataParser is thread-safe. The Resolve* methods may run concurrently with
// each other and with the methods that read and advance the private state
// (ReleaseUnencrypted, SaveRecoveryInfo, RestoreRecoveryInfo), which are
// serialized so that each release token's initial state matches the state that
// KMS stored for the previous release.
class DataParser {
 public:
  DataParser(confidential_federated_compute::Decryptor* blob_decryptor,
             std::string outgoing_server_address,
             std::vector<std::string> reencryption_keys,
             std::string kms_public_key, std::string invocation_id,
             PrivateState* private_state = nullptr,
             std::shared_ptr<oak::crypto::SigningKeyHandle> signing_key_handle =
                 nullptr);

  // Retrieves the TensorProto that is described by the provided blob id and
  // FcCheckpoint key.
  absl::StatusOr<tensorflow_federated::aggregation::TensorProto>
  ResolveBlobIdToTensor(std::string blob_id, std::string key);

  // Retrieves all Tensors stored in the FcCheckpoint described by the
  // provided blob id.
  absl::StatusOr<absl::flat_hash_map<std::string,
                                     tensorflow_federated::aggregation::Tensor>>
  ResolveBlobIdToDict(std::string blob_id);

  // Wraps the data in a release token and sends it to untrusted space.
  absl::Status ReleaseUnencrypted(std::string data, std::string key)
      ABSL_LOCKS_EXCLUDED(private_state_mutex_);

  absl::Status SaveRecoveryInfo(
      std::string recovery_value, std::string recovery_key,
      std::vector<std::pair<std::string, std::string>> release_queue)
      ABSL_LOCKS_EXCLUDED(private_state_mutex_);

  absl::StatusOr<std::string> RestoreRecoveryInfo(std::string recovery_key)
      ABSL_LOCKS_EXCLUDED(private_state_mutex_);

 private:
  absl::Status ReleaseUnencryptedInternal(std::string data, std::string key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(private_state_mutex_);

  Decryptor* blob_decryptor_;
  std::unique_ptr<
      fcp::confidentialcompute::outgoing::DataReadWrite::StubInterface>
      stub_;

  std::string kms_public_key_;
  std::string invocation_id_;

  std::vector<std::string> reencryption_keys_;
  // Serializes the operations that read or advance `private_state_`. It is held
  // across the corresponding Write RPCs so that KMS observes state transitions
  // in the order in which they were created.
  absl::Mutex private_state_mutex_;
  // Private state.
  PrivateState* private_state_ ABSL_PT_GUARDED_BY(private_state_mutex_);
  // The signing key handle used to sign the final result.
  std::shared_ptr<oak::crypto::SigningKeyHandle> signing_key_handle_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_PARSER_H_