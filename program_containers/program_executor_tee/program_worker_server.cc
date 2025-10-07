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
#include "program_executor_tee/program_worker_server.h"

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "cc/oak_session/server_session.h"
#include "fcp/protos/confidentialcompute/program_worker.grpc.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"

namespace confidential_federated_compute::program_executor_tee {

extern "C" {
extern void* enter_tokio_runtime();
extern void exit_tokio_runtime(void* guard_ptr);
}

using ::confidential_federated_compute::program_executor_tee::NoiseLeafExecutor;
using ::fcp::confidentialcompute::ComputationRequest;
using ::fcp::confidentialcompute::ComputationResponse;
using ::grpc::ServerContext;
using ::grpc::Status;

absl::StatusOr<std::unique_ptr<ProgramWorkerTee>> ProgramWorkerTee::Create(
    oak::session::SessionConfig* session_config,
    std::function<
        absl::StatusOr<std::shared_ptr<tensorflow_federated::Executor>>()>
        leaf_executor_factory) {
  FCP_ASSIGN_OR_RETURN(auto server_session,
                       oak::session::ServerSession::Create(session_config));
  FCP_ASSIGN_OR_RETURN(auto noise_leaf_executor,
                       NoiseLeafExecutor::Create(std::move(server_session),
                                                 leaf_executor_factory));
  return absl::WrapUnique(new ProgramWorkerTee(std::move(noise_leaf_executor)));
}

grpc::Status ProgramWorkerTee::Execute(ServerContext* context,
                                       const ComputationRequest* request,
                                       ComputationResponse* response) {
  // Enter the Rust runtime to get the key for session binding.
  void* guard_ptr = enter_tokio_runtime();
  auto cleanup =
      absl::MakeCleanup([guard_ptr] { exit_tokio_runtime(guard_ptr); });

  return noise_leaf_executor_->Execute(request, response);
}

}  // namespace confidential_federated_compute::program_executor_tee
