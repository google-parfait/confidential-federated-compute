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

namespace confidential_federated_compute {

// Class used to track the number of active sessions in a container.
//
// This class is threadsafe.
class SessionTracker {
 public:
  SessionTracker(int max_num_sessions, long max_session_memory_bytes)
      : max_num_sessions_(max_num_sessions),
        max_session_memory_bytes_(max_session_memory_bytes) {};

  // Tries to add a session and returns the amount of memory in bytes that the
  // session is allowed. Returns 0 if there is no available memory.
  long AddSession();

  // Tries to remove a session and returns an error if unable to do so.
  absl::Status RemoveSession();

 private:
  absl::Mutex mutex_;
  int num_sessions_ = 0 ABSL_GUARDED_BY(mutex_);
  int max_num_sessions_;
  // Memory for each session in bytes.
  long max_session_memory_bytes_;
};

}  // namespace confidential_federated_compute
#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SESSION_H_
