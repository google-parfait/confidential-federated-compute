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
#include "containers/program_worker/program_worker_server.h"

#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/program_worker.grpc.pb.h"
#include "fcp/protos/confidentialcompute/program_worker.pb.h"

namespace confidential_federated_compute::program_worker {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ComputationRequest;
using ::fcp::confidentialcompute::ComputationResponse;
using ::grpc::ServerContext;
using ::grpc::Status;

grpc::Status ProgramWorkerTee::Execute(ServerContext* context,
                                       const ComputationRequest* request,
                                       ComputationResponse* response) {
  // TODO - b/378243349: Add implementation based on the computation type.
  return ToGrpcStatus(absl::UnimplementedError("Not implemented"));
}

}  // namespace confidential_federated_compute::program_worker
