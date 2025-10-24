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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_NOISE_LEAF_EXECUTOR_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_NOISE_LEAF_EXECUTOR_H

#include <memory>
#include <utility>

#include "absl/log/die_if_null.h"
#include "cc/oak_session/server_session.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"
#include "grpcpp/support/status.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_service.h"

namespace confidential_federated_compute::program_executor_tee {

// Implementation of a leaf TFF execution stack that receives requests
// through a noise session and returns the responses.
class NoiseLeafExecutor {
 public:
  static absl::StatusOr<std::unique_ptr<NoiseLeafExecutor>> Create(
      std::function<oak::session::SessionConfig*()> session_config_fn,
      std::function<
          absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
          leaf_executor_factory) {
    return absl::WrapUnique(
        new NoiseLeafExecutor(session_config_fn, leaf_executor_factory));
  }

  grpc::Status Execute(
      const fcp::confidentialcompute::ComputationRequest* request,
      fcp::confidentialcompute::ComputationResponse* response);

 private:
  explicit NoiseLeafExecutor(
      std::function<oak::session::SessionConfig*()> session_config_fn,
      std::function<
          absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
          leaf_executor_factory);

  absl::StatusOr<oak::session::v1::PlaintextMessage> DecryptRequest(
      const oak::session::v1::SessionRequest& session_request);

  absl::StatusOr<oak::session::v1::SessionResponse> EncryptResult(
      const oak::session::v1::PlaintextMessage& plaintext_response);

  absl::StatusOr<std::optional<oak::session::v1::SessionResponse>>
  HandleHandshakeRequest(
      const oak::session::v1::SessionRequest& session_request);

  std::unique_ptr<oak::session::ServerSession> server_session_;
  std::unique_ptr<tensorflow_federated::ExecutorService> executor_service_;
  std::function<oak::session::SessionConfig*()> session_config_fn_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_NOISE_LEAF_EXECUTOR_H
