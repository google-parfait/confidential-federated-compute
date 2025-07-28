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

SessionResponse ToSessionWriteFinishedResponse(absl::Status status,
                                               long committed_size_bytes) {
  grpc::Status grpc_status = ToGrpcStatus(std::move(status));
  SessionResponse session_response;
  WriteFinishedResponse* response = session_response.mutable_write();
  response->mutable_status()->set_code(grpc_status.error_code());
  response->mutable_status()->set_message(grpc_status.error_message());
  response->set_committed_size_bytes(committed_size_bytes);
  return session_response;
}

SessionResponse ToSessionCommitResponse(
    absl::Status status, int num_inputs_committed,
    std::vector<absl::Status> ignored_errors) {
  grpc::Status grpc_status = ToGrpcStatus(std::move(status));
  SessionResponse session_response;
  CommitResponse* response = session_response.mutable_commit();
  response->mutable_status()->set_code(grpc_status.error_code());
  response->mutable_status()->set_message(grpc_status.error_message());
  response->mutable_stats()->set_num_inputs_committed(num_inputs_committed);
  for (absl::Status& ignored_error : ignored_errors) {
    grpc::Status grpc_ignored_error = ToGrpcStatus(std::move(ignored_error));
    auto* ignored_status = response->mutable_stats()->add_ignored_errors();

    ignored_status->set_code(grpc_ignored_error.error_code());
    ignored_status->set_message(grpc_ignored_error.error_message());
  }
  return session_response;
}

absl::StatusOr<fcp::confidentialcompute::ConfigureResponse>
LegacySession::Configure(fcp::confidentialcompute::ConfigureRequest request,
                         Context& context) {
  SessionRequest session_request;
  *session_request.mutable_configure() = std::move(request);
  FCP_RETURN_IF_ERROR(ConfigureSession(std::move(session_request)));
  return ConfigureResponse{};
}

absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse>
LegacySession::Write(fcp::confidentialcompute::WriteRequest request,
                     std::string unencrypted_data, Context& context) {
  FCP_ASSIGN_OR_RETURN(SessionResponse response,
                       SessionWrite(request, std::move(unencrypted_data)));
  return std::move(*response.mutable_write());
}

absl::StatusOr<fcp::confidentialcompute::CommitResponse> LegacySession::Commit(
    fcp::confidentialcompute::CommitRequest request, Context& context) {
  FCP_ASSIGN_OR_RETURN(SessionResponse response, SessionCommit(request));
  return std::move(*response.mutable_commit());
}

absl::StatusOr<fcp::confidentialcompute::FinalizeResponse>
LegacySession::Finalize(fcp::confidentialcompute::FinalizeRequest request,
                        fcp::confidentialcompute::BlobMetadata input_metadata,
                        Context& context) {
  FCP_ASSIGN_OR_RETURN(SessionResponse session_response,
                       FinalizeSession(request, input_metadata));
  FCP_CHECK(session_response.has_read());

  FinalizeResponse finalize_response;
  if (session_response.read()
          .first_response_configuration()
          .Is<FinalResultConfiguration>()) {
    FinalResultConfiguration final_result_configuration;
    session_response.read().first_response_configuration().UnpackTo(
        &final_result_configuration);
    finalize_response.set_release_token(
        std::move(*final_result_configuration.mutable_release_token()));
    *finalize_response.mutable_configuration() = std::move(
        *final_result_configuration.mutable_application_configuration());
  }

  if (!context.Emit(*std::move(session_response).mutable_read())) {
    return absl::AbortedError("Failed to write ReadResponse to the stream.");
  }
  return finalize_response;
}

}  // namespace confidential_federated_compute
