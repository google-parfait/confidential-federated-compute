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

// This file contains functions and classes for managing aggregation sessions
// of a ConfidentialTransform service.
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SESSION_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SESSION_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute {

// Class used to track the number of active sessions in a container.
//
// This class is threadsafe.
class SessionTracker {
 public:
  SessionTracker(int max_num_sessions)
      : available_sessions_(max_num_sessions),
        max_num_sessions_(max_num_sessions) {};

  // Tries to add a session and returns the amount of memory in bytes that the
  // session is allowed. Returns 0 if there is no available memory.
  absl::Status AddSession();

  // Tries to remove a session and returns an error if unable to do so.
  absl::Status RemoveSession();

 private:
  absl::Mutex mutex_;
  int available_sessions_ ABSL_GUARDED_BY(mutex_) = 0;
  int max_num_sessions_;
};

// Create a SessionResponse with a WriteFinishedResponse.
fcp::confidentialcompute::SessionResponse ToSessionWriteFinishedResponse(
    absl::Status status, long committed_size_bytes = 0);

// Interface for interacting with a session in a container. Implementations
// may not be threadsafe.
class Session {
 public:
  // Initialize the session with the given configuration.
  virtual absl::Status ConfigureSession(
      fcp::confidentialcompute::SessionRequest configure_request) = 0;
  // Incorporate a write request into the session.
  virtual absl::StatusOr<fcp::confidentialcompute::SessionResponse>
  SessionWrite(const fcp::confidentialcompute::WriteRequest& write_request,
               std::string unencrypted_data) = 0;
  // Run any session finalization logic and complete the session.
  // After finalization, the session state is no longer mutable.
  virtual absl::StatusOr<fcp::confidentialcompute::SessionResponse>
  FinalizeSession(
      const fcp::confidentialcompute::FinalizeRequest& request,
      const fcp::confidentialcompute::BlobMetadata& input_metadata) = 0;

  virtual ~Session() = default;
};

}  // namespace confidential_federated_compute
#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SESSION_H_
