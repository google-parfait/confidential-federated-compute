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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINTERS_BATCHED_INFERENCE_BATCHED_INFERENCE_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINTERS_BATCHED_INFERENCE_BATCHED_INFERENCE_SERVER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "cc/containers/sdk/encryption_key_handle.h"
#include "cc/containers/sdk/signing_key_handle.h"
#include "containers/batched_inference/batched_inference_provider.h"
#include "containers/fns/fn_factory.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::batched_inference {

using oak::crypto::EncryptionKeyHandle;
using oak::crypto::SigningKeyHandle;

class BatchedInferenceServer {
 public:
  virtual ~BatchedInferenceServer() {}
  virtual void Wait() = 0;
  virtual int port() = 0;
};

absl::StatusOr<std::unique_ptr<BatchedInferenceServer>>
CreateBatchedInferenceServer(
    std::shared_ptr<BatchedInferenceProvider> batched_inference_provider,
    int incoming_port, std::unique_ptr<SigningKeyHandle> signing_handle,
    std::unique_ptr<EncryptionKeyHandle> encryption_handle);

}  // namespace confidential_federated_compute::batched_inference

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINTERS_BATCHED_INFERENCE_BATCHED_INFERENCE_SERVER_H_
