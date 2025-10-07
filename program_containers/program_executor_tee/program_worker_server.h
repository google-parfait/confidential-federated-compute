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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_WORKER_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_WORKER_SERVER_H_

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cc/oak_session/config.h"
#include "fcp/protos/confidentialcompute/program_worker.grpc.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"
#include "grpcpp/support/status.h"
#include "program_executor_tee/program_context/cc/noise_leaf_executor.h"

namespace confidential_federated_compute::program_executor_tee {

// ProgramWorker service implementation.
class ProgramWorkerTee
    : public fcp::confidentialcompute::ProgramWorker::Service {
 public:
  static absl::StatusOr<std::unique_ptr<ProgramWorkerTee>> Create(
      oak::session::SessionConfig* session_config,
      std::function<
          absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
          leaf_executor_factory);

  grpc::Status Execute(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::ComputationRequest* request,
      fcp::confidentialcompute::ComputationResponse* response);

 private:
  explicit ProgramWorkerTee(
      std::unique_ptr<confidential_federated_compute::program_executor_tee::
                          NoiseLeafExecutor>
          noise_leaf_executor)
      : noise_leaf_executor_(std::move(noise_leaf_executor)) {};

  std::unique_ptr<
      confidential_federated_compute::program_executor_tee::NoiseLeafExecutor>
      noise_leaf_executor_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_WORKER_SERVER_H_
