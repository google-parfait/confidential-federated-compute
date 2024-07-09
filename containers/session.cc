// Copyright 2024 Google LLC.
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
#include "containers/session.h"

#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;

long SessionTracker::AddSession() {
  absl::MutexLock l(&mutex_);
  if (num_sessions_ < max_num_sessions_) {
    num_sessions_++;
    return max_session_memory_bytes_;
  }
  return 0;
}

absl::Status SessionTracker::RemoveSession() {
  absl::MutexLock l(&mutex_);
  if (num_sessions_ <= 0) {
    return absl::FailedPreconditionError(
        "SessionTracker: no sessions to remove.");
  }
  num_sessions_--;
  return absl::OkStatus();
}

SessionResponse ToSessionWriteFinishedResponse(absl::Status status,
                                               long available_memory,
                                               long committed_size_bytes) {
  grpc::Status grpc_status = ToGrpcStatus(std::move(status));
  SessionResponse session_response;
  WriteFinishedResponse* response = session_response.mutable_write();
  response->mutable_status()->set_code(grpc_status.error_code());
  response->mutable_status()->set_message(grpc_status.error_message());
  response->set_write_capacity_bytes(available_memory);
  response->set_committed_size_bytes(committed_size_bytes);
  return session_response;
}

}  // namespace confidential_federated_compute
