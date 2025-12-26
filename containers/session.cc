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

#include <filesystem>

#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "grpcpp/support/status.h"

namespace confidential_federated_compute {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::FinalizeResponse;
using ::fcp::confidentialcompute::FinalResultConfiguration;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;

constexpr absl::Duration kAddSessionPreconditionTimeout = absl::Seconds(1);

absl::Status SessionTracker::AddSession() {
  absl::MutexLock l(&mutex_);
  // Wait for the nonzero number of available sessions in case
  // there is a race between creating a new session and destroying
  // an old session.
  mutex_.AwaitWithTimeout(
      absl::Condition(
          +[](int* available) { return *available > 0; }, &available_sessions_),
      kAddSessionPreconditionTimeout);
  if (available_sessions_ > 0) {
    available_sessions_--;
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      "SessionTracker: already at the maximum number of sessions.");
}

absl::Status SessionTracker::RemoveSession() {
  absl::MutexLock l(&mutex_);
  if (available_sessions_ >= max_num_sessions_) {
    return absl::FailedPreconditionError(
        "SessionTracker: no sessions to remove.");
  }
  available_sessions_++;
  return absl::OkStatus();
}

WriteFinishedResponse ToWriteFinishedResponse(absl::Status status,
                                              long committed_size_bytes) {
  grpc::Status grpc_status = ToGrpcStatus(std::move(status));
  WriteFinishedResponse response;
  response.mutable_status()->set_code(grpc_status.error_code());
  response.mutable_status()->set_message(grpc_status.error_message());
  response.set_committed_size_bytes(committed_size_bytes);
  return response;
}

SessionResponse ToSessionWriteFinishedResponse(absl::Status status,
                                               long committed_size_bytes) {
  SessionResponse session_response;
  *session_response.mutable_write() =
      ToWriteFinishedResponse(std::move(status), committed_size_bytes);
  return session_response;
}

CommitResponse ToCommitResponse(absl::Status status, int num_inputs_committed,
                                std::vector<absl::Status> ignored_errors) {
  grpc::Status grpc_status = ToGrpcStatus(std::move(status));
  CommitResponse response;
  response.mutable_status()->set_code(grpc_status.error_code());
  response.mutable_status()->set_message(grpc_status.error_message());
  response.mutable_stats()->set_num_inputs_committed(num_inputs_committed);
  for (absl::Status& ignored_error : ignored_errors) {
    grpc::Status grpc_ignored_error = ToGrpcStatus(std::move(ignored_error));
    auto* ignored_status = response.mutable_stats()->add_ignored_errors();

    ignored_status->set_code(grpc_ignored_error.error_code());
    ignored_status->set_message(grpc_ignored_error.error_message());
  }
  return response;
}

SessionResponse ToSessionCommitResponse(
    absl::Status status, int num_inputs_committed,
    std::vector<absl::Status> ignored_errors) {
  SessionResponse session_response;
  *session_response.mutable_commit() = ToCommitResponse(
      std::move(status), num_inputs_committed, std::move(ignored_errors));
  return session_response;
}

}  // namespace confidential_federated_compute
