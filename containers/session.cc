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

namespace confidential_federated_compute {

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

}  // namespace confidential_federated_compute
